from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from quick_convert.components.encoders.conformer_encoder import ConformerEncoderSSL
from quick_convert.utils.masking import make_padding_mask, masked_loss, trim_to_min

from .speaker_head import SpeakerASPHead
from .linguistic_head import LinguisticCTCHead
from .linear_head import LinearHead

from ..layers import ResidualVectorQuantizer, GradientReversalLayer, VectorQuantize


class RVQLayerRouter(nn.Module):
    """
    Routes the output of the RVQ to different heads for disentanglement.
    """

    def __init__(self, n_classes: int, codebook_dim: int, codebook_size: int, gumbel_tau: float = 1.0):
        super().__init__()
        self.n_classes = n_classes
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.gumbel_tau = gumbel_tau

        # Learnable class embeddings for routing
        # output shape: (n_classes, 1)
        self.classifier = nn.Linear(codebook_size, n_classes)

    def _compute_mask(
        self,
        quantizers: List[VectorQuantize],
        compute_loss: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
            quantizers: List of VectorQuantize modules from the RVQ, length = num_codebooks.
            compute_loss: If True, compute and return the load-balancing loss.

        Returns:
            layer_mask: Hard routing mask, shape (num_codebooks, n_classes).
            loss: Load-balancing loss if compute_loss=True, otherwise None.
        """

        # Always compute probabilities/logits, even if we may reuse a cached eval mask,
        # because compute_loss=True needs layer_probabilities.
        # detach the codebook weights to avoid backprop through the quantizers
        weights = torch.stack(
            [q.codebook.weight.mean(dim=1).detach() for q in quantizers],
            dim=0,
        )  # (num_codebooks, codebook_dim)

        layer_logits = self.classifier(weights)  # (num_codebooks, n_classes)
        layer_probabilities = F.softmax(layer_logits, dim=-1)

        cached_mask = getattr(self, "layer_mask", None)

        if self.training:
            # Training: always sample a fresh hard mask with straight-through gradients.
            layer_mask = F.gumbel_softmax(
                layer_logits,
                tau=self.gumbel_tau,
                hard=True,
                dim=-1,
            )

            # Don't detach in training; the straight-through mask participates in routing.
            self.layer_mask = layer_mask

        else:
            if cached_mask is not None:
                # Eval/inference: reuse the cached deterministic mask.
                layer_mask = cached_mask.to(
                    device=layer_logits.device,
                    dtype=layer_logits.dtype,
                )
            else:
                # Eval/inference with no cache: make deterministic argmax mask.
                selected = torch.argmax(layer_logits, dim=-1)
                layer_mask = torch.zeros_like(layer_probabilities).scatter_(
                    1,
                    selected.unsqueeze(-1),
                    1.0,
                )

                self.layer_mask = layer_mask.detach()

        if compute_loss:
            loss = self.compute_loss(layer_probabilities, layer_mask)
        else:
            loss = None

        return layer_mask, loss

    def forward(
        self, quantizers: List[VectorQuantize], z_quantized: List[torch.Tensor], compute_loss: bool = False
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Args:
            quantizers: List of VectorQuantize modules from the RVQ, length = num_codebooks.
            z_quantized: List of quantized outputs from each codebook, each of shape (B·T, F).

        Returns:
            z_spk: Disentangled speaker features, shape (B, T, F).
            z_ling: Disentangled linguistic features, shape (B, T, F).
            z_pros: Disentangled prosody features, shape (B, T, F) or None if no prosody head.
            router_loss: Load balancing loss for the router, scalar tensor.
        """

        layer_mask, loss = self._compute_mask(quantizers, compute_loss=compute_loss)
        z_quantized = torch.stack(z_quantized, dim=0)  # (num_codebooks, B, F, T)

        # Get each class's selected layers and sum them to get the representation.
        # z_quantized: (num_codebooks, B, F, T)
        # layer_mask[:, n]: (num_codebooks,)
        z_s = []
        for n in range(self.n_classes):
            z_n = (z_quantized * layer_mask[:, n].view(-1, 1, 1, 1)).sum(dim=0)  # (B, F, T)
            z_s.append(z_n)

        return z_s, loss

    def compute_loss(self, layer_probabilities: torch.Tensor, one_hot_mask: torch.Tensor) -> torch.Tensor:
        """
        Computes load balancing loss to promote more uniform codebook usage.

        Args:
            layer_probabilities: Probabilities of each layer being selected for each class, shape (num_codebooks, n_classes).
            one_hot_mask: One-hot encoded mask of selected layers for each class, shape (num_codebooks, n_classes).

        Returns:
            router_loss: Load balancing loss for the router, scalar tensor.
        """
        # P_j: average router probability mass sent to class j.
        # f_j: fraction of layers hard-routed to class j.
        p_expert = layer_probabilities.mean(dim=0)  # (n_classes,)
        f_expert = one_hot_mask.float().mean(dim=0)  # (n_classes,)

        # Load-balancing objective: n_classes * sum_j f_j * P_j.
        # Lower is better; the minimum (1.0) is achieved when both distributions are uniform.
        loss = self.n_classes * torch.sum(f_expert * p_expert)
        return loss


class RVQDisentangler(nn.Module):
    """
    Disentanglement model with three components:

      1. W2VBertContentEncoder — waveform (B, T) -> (B, T, L, F)
      2. ParallelConformerEncoder (content encoder) — (B, T, L, F) -> (B, T, F)
      3. ResidualVectorQuantizer (RVQ) — (B·T, F) -> z_q, indices, perplexity, remainders

    Two entries of the RVQ remainder list are selected at the indices given by
    ``rvq_spk_idx`` and ``rvq_pros_idx``.  ``remainder_list[0]`` is the
    original content vector; ``remainder_list[k]`` (k ≥ 1) is the residual
    after the k-th codebook.

    Args:
        feature_extractor:  Instantiated W2VBertContentEncoder.
        content_encoder:    Instantiated ParallelConformerEncoder.
        rvq:                Instantiated ResidualVectorQuantizer.
        rvq_spk_idx:        Index into remainder_list for speaker features.
        rvq_pros_idx:       Index into remainder_list for prosody features.
    """

    def __init__(
        self,
        # feature_extractor: W2VBertContentEncoder,  # TODO - ideally this would be an interface that could support multiple SSL models
        content_encoder: ConformerEncoderSSL,
        rvq: ResidualVectorQuantizer,
        linguistic_head: LinguisticCTCHead,
        speaker_head: SpeakerASPHead,
        emotion_head: LinearHead,
        prosody_head: LinearHead | None,
        router: dict[str, int] | RVQLayerRouter = {"content": 0, "speaker": 1, "prosody": 2, "emotion": 2},
        *kwargs,
    ):
        super().__init__()

        self.content_encoder = content_encoder
        self.rvq = rvq

        self.speaker_head = speaker_head
        self.linguistic_head = linguistic_head
        self.emotion_head = emotion_head
        self.prosody_head = prosody_head

        if isinstance(router, dict):
            valid_keys = {"content", "speaker", "prosody", "emotion"}
            given_keys = set(router.keys())
            assert given_keys == valid_keys, f"keys in router dict must be exactly: {valid_keys}, got: {given_keys}"

        self.router = router
        self._create_adversarial_heads()

    def _create_adversarial_heads(self):
        self.adv_speaker_head_ling = copy.deepcopy(self.speaker_head)  # For adversarial loss on speaker features
        self.adv_speaker_head_pros = copy.deepcopy(self.speaker_head)  # For adversarial loss on speaker features
        self.adv_linguistic_head_spk = copy.deepcopy(
            self.linguistic_head
        )  # For adversarial loss on linguistic features
        self.adv_linguistic_head_pros = copy.deepcopy(
            self.linguistic_head
        )  # For adversarial loss on linguistic features
        self.grl = GradientReversalLayer()  # For adversarial loss on speaker features

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> List[torch.Tensor]:
        with torch.no_grad():
            z_q, z_quantized, spk_q, text_q, emo_pros_q = self.encode(features, lengths)[0:5]
            spk_output = self.speaker_head(spk_q)

            return z_q, z_quantized, text_q, spk_q, spk_output, emo_pros_q

    def encode(
        self,
        features: torch.Tensor,  # (B, T_samples)
        padding_mask: torch.Tensor,  # (B,)  — number of valid samples per item
    ) -> Tuple[
        torch.Tensor,  # z_q            (B, T, F)
        List[torch.Tensor],  # indices_list   [num_codebooks × (B·T,)]
        List[float],  # perplexity_list
        List[torch.Tensor],  # remainder_list (flat, length = num_codebooks + 1)
        torch.Tensor,  # spk_remainder  (B, T, F)
        torch.Tensor | None,  # pros_remainder (B, T, F)
        torch.Tensor | None,  # emo_remainder  (B, T, F)
    ]:
        """
        Args:
            features: Extracted content features of shape (B, T_samples).
            lengths:  Number of valid content features per batch item, shape (B,).

        Returns:
            z_q:             Differentiable RVQ reconstruction, shape (B, T, F).
            text_quantized:  Disentangled linguistic features, shape (B, T, F).
            spk_quantized:   Disentangled speaker features, shape (B, T, F).
            pros_quantized:  Disentangled prosody features, shape (B, T, F) or None if no prosody head.
            emo_quantized:   Disentangled emotion features, shape (B, T, F) or None if no emotion head.
            commitment_loss: Commitment loss from the RVQ, scalar tensor.
            codebook_loss:   Codebook loss from the RVQ, scalar tensor.
            lengths:         Updated lengths after feature extraction, shape (B,).
        """

        # 4. Content encoder: (B, T, L, F) -> (B, T, F)
        content = self.content_encoder(features, padding_mask=padding_mask)

        # 5. Flatten temporal dim for the RVQ which expects (N, D)
        B, T, F = content.shape
        content = content.transpose(1, 2)  # (B, F, T)

        # 6. Residual VQ
        # z_q, z_qs, codes, latents, commitment_loss, codebook_loss
        z_q_flat, z_quantized, _, _, commitment_loss, codebook_loss = self.rvq(content, padding_mask)

        # 7. Reshape back to (B, T, F)
        z_q = z_q_flat.transpose(1, 2)  # (B, T, F)
        z_spk, z_ling, z_pros, router_loss = self._route(z_quantized)

        return (
            z_q,
            z_quantized,
            z_spk,
            z_ling,
            z_pros,
            commitment_loss,
            codebook_loss,
            router_loss,
            content,
        )

    def _route(self, z_quantized):

        if isinstance(self.router, dict):
            # 8. Select the disentangled representations
            z_spk = z_quantized[self.router["speaker"]].transpose(1, 2)  # (B, T, F)
            # TODO: rename to linguistic content
            z_ling = z_quantized[self.router["content"]].transpose(1, 2)  # (B, T, F)

            # For prosody and emotion we sum the remaining quantized vectors, giving the RVQ
            # the flexibility to decide how to allocate information across codebooks
            z_pros = torch.stack(z_quantized[self.router["emo_pros"] :], dim=3).sum(dim=3).transpose(1, 2)  # (B, T, F)
            router_loss = z_quantized.new_tensor(0.0)

        else:
            routed_z_s, router_loss = self.router(self.rvq.quantizers, z_quantized, compute_loss=True)
            z_spk = routed_z_s[0].transpose(1, 2)  # (B, T, F)
            z_ling = routed_z_s[1].transpose(1, 2)  # (B, T, F)
            z_pros = routed_z_s[2].transpose(1, 2)  # (B, T, F

        return (z_spk, z_ling, z_pros, router_loss)

    def _shared_step(self, features, lengths, emotion_seq, emotion_lengths):
        # DAC content frames vs emo2vec frames differ by a couple (different framing
        # conventions across model families), so allow a looser tolerance than the
        # default 1 used for same-family (w2v-bert / emo2vec) pairs.
        features, emotion_seq, lengths = trim_to_min(
            features, emotion_seq, lengths, emotion_lengths, time_dim=1, max_diff=4
        )
        padding_mask = make_padding_mask(lengths, max_length=features.shape[1])  # (B, T_frames)
        return *self.encode(features, padding_mask), padding_mask

    def compute_loss(
        self,
        features: int["b t d_feat"],
        lengths: int["b"],
        linguistic_targets: int["b t_txt"],
        target_lengths: int["b"],
        speaker_seq: float["b d_spk"] | int["b"],
        emotion_seq: Optional[float["b t d_emo"]],
        emotion_lengths: Optional[int["b"]],
        # the target speaker indices encoded from their string ids, e.g. `16`, not `spk11`
        # speaker_targets: Optional[int["b"]] = None,
        prosody_seq: Optional[float["b t d_pro"]] = None,
        run_adv=True,
    ) -> List:

        z_q, z_quantized, z_spk, z_ling, z_pros, commitment_loss, codebook_loss, router_loss, content, padding_mask = (
            self._shared_step(features, lengths, emotion_seq, emotion_lengths)
        )

        # MSE loss between RVQ output and content encoder output
        # to encourage the RVQ to capture the all of the information from the content encoder
        rvq_mse_loss = masked_loss(
            F.mse_loss, z_q, content.detach().transpose(1, 2), padding_mask
        )  # content is (B, F, T), z_q is (B, T, F) after transpose

        rvq_losses = {
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss,
            "mse_loss": rvq_mse_loss,
            "load_balancing_loss": router_loss,
        }

        # Speaker loss: encourage spk_q to match the target speaker embedding
        spk_output, spk_loss, spk_acc, _ = self.speaker_head.compute_loss(z_spk, speaker_seq, padding_mask)

        # Emotion loss: encourage emo_q to match the target emotion features (if provided)
        emo_loss = self.emotion_head.compute_loss(z_pros, emotion_seq, padding_mask)

        # Prosody loss: encourage pros_q to match the target prosody features (if provided)
        if self.prosody_head is not None and prosody_seq is not None:
            pros_loss = self.prosody_head.compute_loss(z_pros, prosody_seq)
        else:
            pros_loss = 0.0

        # CTC loss: encourage text_q to predict the target linguistic sequence
        ctc_loss = self.linguistic_head.compute_loss(
            z_ling,
            linguistic_targets,
            input_lengths=lengths,
            target_lengths=target_lengths,
        )

        distill_losses = {
            "ctc_loss": ctc_loss,
            "spk_loss": spk_loss,
            "pros_loss": pros_loss,
            "emo_loss": emo_loss,
        }

        if run_adv:
            # Adversarial speaker loss over linguistic features: encourage spk_q to be uninformative about speaker identity
            _, adv_spk_loss_ling, adv_spk_acc_ling, _ = self.adv_speaker_head_ling.compute_loss(
                self.grl(z_ling), speaker_seq
            )

            # Adversarial speaker loss over prosody features: encourage pros_q to be uninformative about speaker identity
            _, adv_spk_loss_pros, adv_spk_acc_pros, _ = self.adv_speaker_head_pros.compute_loss(
                self.grl(z_pros), speaker_seq
            )

            # Adversarial linguistic loss over speaker features: encourage text_q to be uninformative about linguistic content
            adv_ling_loss_spk = self.adv_linguistic_head_spk.compute_loss(
                self.grl(z_spk),
                linguistic_targets,
                input_lengths=lengths,
                target_lengths=target_lengths,
            )

            # Adversarial linguistic loss over prosody features: encourage text_q to be uninformative about linguistic content
            adv_ling_loss_pros = self.adv_linguistic_head_pros.compute_loss(
                self.grl(z_pros),
                linguistic_targets,
                input_lengths=lengths,
                target_lengths=target_lengths,
            )
        else:
            (
                adv_spk_loss_ling,
                adv_spk_loss_pros,
                adv_ling_loss_spk,
                adv_ling_loss_pros,
                adv_spk_acc_ling,
                adv_spk_acc_pros,
            ) = (0,) * 6

        adv_losses = {
            "adv_spk_loss_ling": adv_spk_loss_ling,
            "adv_spk_loss_pros": adv_spk_loss_pros,
            "adv_ling_loss_spk": adv_ling_loss_spk,
            "adv_ling_loss_pros": adv_ling_loss_pros,
        }

        loss_dict = {
            "rvq_losses": rvq_losses,
            "distill_losses": distill_losses,
            "adv_losses": adv_losses,
        }

        # TODO - fix method output to return spk_acc_dict
        spk_acc_dict = {
            "spk_acc": spk_acc,
            "adv_spk_acc_ling": adv_spk_acc_ling,
            "adv_spk_acc_pros": adv_spk_acc_pros,
        }

        return [z_quantized, z_spk, spk_output, z_ling, z_pros, loss_dict, spk_acc_dict, lengths]

    def inference(
        self,
        features: int["b t d_feat"],
        lengths: int["b"],
        emotion_seq: Optional[float["b t d_emo"]],
        emotion_lengths: Optional[int["b"]],
        # the target speaker indices encoded from their string ids, e.g. `16`, not `spk11`
        # speaker_targets: Optional[int["b"]] = None,
        prosody_seq: Optional[float["b t d_pro"]] = None,
    ):
        features, emotion_seq, lengths = trim_to_min(
            features, emotion_seq, lengths, emotion_lengths, time_dim=1, max_diff=4
        )
        padding_mask = make_padding_mask(lengths, max_length=features.shape[1])  # (B, T_frames)
        z_q, z_quantized, z_spk, z_ling, z_pros, *_, content, padding_mask = self.encode(features, padding_mask)
        spk_output = self.speaker_head(z_spk, padding_mask=padding_mask)
        return z_q, z_quantized, z_spk, spk_output, z_ling, z_pros, content, padding_mask

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from quick_convert.components.encoders.conformer_encoder import ConformerEncoderSSL
from quick_convert.components.layers.rvq import RVQOutput
from quick_convert.utils.masking import make_padding_mask, masked_loss, trim_to_min

from .speaker_head import SpeakerASPHead, SpeakerASRHeadOutput
from .linguistic_head import LinguisticCTCHead
from .linear_head import LinearHead

from ..layers import ResidualVectorQuantizer, GradientReversalLayer, VectorQuantize


@dataclass
class RouterOutput:
    zs: list[torch.Tensor]
    layer_mask: Optional[torch.Tensor] = None
    layer_probabilities: Optional[torch.Tensor] = None
    layer_logits: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None


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
        self.classifier = nn.Linear(codebook_size, n_classes)

    def _compute_mask(
        self,
        quantizers: List[VectorQuantize],
        compute_loss: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        weights = torch.stack(
            [q.codebook.weight.mean(dim=1).detach() for q in quantizers],
            dim=0,
        )

        layer_logits = self.classifier(weights)
        layer_probabilities = F.softmax(layer_logits, dim=-1)

        cached_mask = getattr(self, "layer_mask", None)

        if self.training:
            layer_mask = F.gumbel_softmax(
                layer_logits,
                tau=self.gumbel_tau,
                hard=True,
                dim=-1,
            )
            self.layer_mask = layer_mask
        else:
            if cached_mask is not None:
                layer_mask = cached_mask.to(
                    device=layer_logits.device,
                    dtype=layer_logits.dtype,
                )
            else:
                selected = torch.argmax(layer_logits, dim=-1)
                layer_mask = torch.zeros_like(layer_probabilities).scatter_(
                    1,
                    selected.unsqueeze(-1),
                    1.0,
                )
                self.layer_mask = layer_mask.detach()

        if compute_loss:
            router_loss = self.compute_loss(layer_probabilities, layer_mask)
        else:
            router_loss = None

        return layer_mask, router_loss, layer_probabilities, layer_logits

    def forward(
        self, quantizers: List[VectorQuantize], z_quantized: List[torch.Tensor], compute_loss: bool = False
    ) -> RouterOutput:
        layer_mask, router_loss, layer_probabilities, layer_logits = self._compute_mask(
            quantizers, compute_loss=compute_loss
        )
        z_quantized = torch.stack(z_quantized, dim=0)

        z_s = []
        for n in range(self.n_classes):
            z_n = (z_quantized * layer_mask[:, n].view(-1, 1, 1, 1)).sum(dim=0)
            z_s.append(z_n)

        return RouterOutput(
            zs=z_s,
            layer_mask=layer_mask,
            layer_probabilities=layer_probabilities,
            layer_logits=layer_logits,
            loss=router_loss,
        )

    def compute_loss(self, layer_probabilities: torch.Tensor, one_hot_mask: torch.Tensor) -> torch.Tensor:
        p_expert = layer_probabilities.mean(dim=0)
        f_expert = one_hot_mask.float().mean(dim=0)
        loss = self.n_classes * torch.sum(f_expert * p_expert)
        return loss


@dataclass
class RVQDisentanglerLoss:
    rvq: dict[str, torch.Tensor]
    distill: dict[str, torch.Tensor]
    adv: Optional[dict[str, torch.Tensor]] = None
    metrics: Optional[dict[str, torch.Tensor]] = None
    states: Optional[dict[str, torch.Tensor]] = None


@dataclass
class RVQDisentanglerOutput:
    content: torch.Tensor

    # having both lengths and padding mask is techincally redundant, but we will need different ones for different
    # purposes and it makes more sense to store both than to constantly be re-computing one from the other.
    lengths: torch.Tensor
    padding_mask: torch.Tensor

    rvq: RVQOutput
    router: RouterOutput

    # # head inputs
    # z_spk: torch.Tensor
    # z_ling: torch.Tensor
    # z_pros: torch.Tensor

    # the head output exists on the disentangler (encoder) level. It's akin to an x-vector
    head_outputs: Optional[dict[str, Any]] = None
    loss: Optional[RVQDisentanglerLoss] = None


class RVQDisentangler(nn.Module):
    def __init__(
        self,
        content_encoder: ConformerEncoderSSL,
        rvq: ResidualVectorQuantizer,
        linguistic_head: LinguisticCTCHead,
        speaker_head: SpeakerASPHead,
        emotion_head: LinearHead,
        prosody_head: LinearHead | None,
        router: dict[str, int] | RVQLayerRouter = {"linguistic_content": 0, "speaker": 1, "prosody": 2, "emotion": 2},
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
            valid_keys = {"linguistic_content", "speaker", "prosody", "emotion"}
            given_keys = set(router.keys())
            assert given_keys == valid_keys, f"keys in router dict must be exactly: {valid_keys}, got: {given_keys}"

        self.router = router
        self._create_adversarial_heads()

    def _create_adversarial_heads(self):
        self.adv_speaker_head_ling = copy.deepcopy(self.speaker_head)
        self.adv_speaker_head_pros = copy.deepcopy(self.speaker_head)
        self.adv_linguistic_head_spk = copy.deepcopy(self.linguistic_head)
        self.adv_linguistic_head_pros = copy.deepcopy(self.linguistic_head)
        self.grl = GradientReversalLayer()

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> RVQDisentanglerOutput:
        with torch.no_grad():
            return self.encode(features, lengths)

    def encode(
        self,
        features: int["b t d"],
        padding_mask: int["b t [1]"],
        # cursory addition, because the output of this function should
        lengths: Optional[int["b"]] = None,
    ) -> RVQDisentanglerOutput:
        content = self.content_encoder(features, padding_mask=padding_mask)

        B, T, F = content.shape
        content = content.transpose(1, 2)

        rvq_output: RVQOutput = self.rvq(content, padding_mask)

        rvq_output = replace(rvq_output, z_q=rvq_output.z_q.transpose(1, 2))
        router_output = self._route(rvq_output.layer_z_qs)

        return RVQDisentanglerOutput(
            content=content,
            # having lengths and padding
            lengths=lengths if lengths is not None else padding_mask.sum(dim=1),
            padding_mask=padding_mask,
            rvq=rvq_output,
            router=router_output,
        )

    def _route(self, layer_z_qs) -> RouterOutput:
        if isinstance(self.router, dict):
            z_spk = layer_z_qs[self.router["speaker"]].transpose(1, 2)
            z_ling = layer_z_qs[self.router["linguistic_content"]].transpose(1, 2)

            z_pros = torch.stack(layer_z_qs[self.router["emo_pros"] :], dim=3).sum(dim=3).transpose(1, 2)
            return RouterOutput(zs=[z_spk, z_ling, z_pros], router_loss=layer_z_qs[0].new_tensor(0.0))

        router_output = self.router(self.rvq.quantizers, layer_z_qs, compute_loss=True)
        z_spk = router_output.zs[0].transpose(1, 2)
        z_ling = router_output.zs[1].transpose(1, 2)
        z_pros = router_output.zs[2].transpose(1, 2)

        return replace(router_output, zs=[z_spk, z_ling, z_pros])

    def _shared_step(self, features, lengths, emotion_seq, emotion_lengths):
        features, emotion_seq, lengths = trim_to_min(
            features, emotion_seq, lengths, emotion_lengths, time_dim=1, max_diff=4
        )
        padding_mask = make_padding_mask(lengths, max_length=max(features.shape[1], emotion_seq.shape[1]))
        return self.encode(features, padding_mask, lengths=lengths)

    def compute_loss(
        self,
        features,
        lengths,
        linguistic_targets,
        target_lengths,
        speaker_seq,
        emotion_seq,
        emotion_lengths,
        prosody_seq=None,
        run_adv=True,
    ) -> RVQDisentanglerOutput:
        output = self._shared_step(features, lengths, emotion_seq, emotion_lengths)

        rvq_mse_loss = masked_loss(
            F.mse_loss,
            preds=output.rvq.z_q,
            targets=output.content.detach().transpose(1, 2),
            mask=output.padding_mask,
        )

        rvq_losses = {
            "commitment_loss": output.rvq.loss.commitment_loss,
            "codebook_loss": output.rvq.loss.codebook_loss,
            "mse_loss": rvq_mse_loss,
            "load_balancing_loss": output.router.loss,
        }

        z_spk, z_ling, z_pros = output.router.zs
        spk_output = self.speaker_head.compute_loss(z_spk, speaker_seq, output.padding_mask)
        emo_loss = self.emotion_head.compute_loss(z_pros, emotion_seq, output.padding_mask)

        if self.prosody_head is not None and prosody_seq is not None:
            pros_loss = self.prosody_head.compute_loss(z_pros, prosody_seq)
        else:
            pros_loss = 0.0

        ctc_loss = self.linguistic_head.compute_loss(
            z_ling,
            linguistic_targets,
            input_lengths=lengths,
            target_lengths=target_lengths,
        )

        distill_losses = {
            "ctc_loss": ctc_loss,
            "spk_loss": spk_output.loss,
            "pros_loss": pros_loss,
            "emo_loss": emo_loss,
        }

        if run_adv:
            _, adv_spk_loss_ling, adv_spk_acc_ling, _ = self.adv_speaker_head_ling.compute_loss(
                self.grl(z_ling), speaker_seq
            )

            _, adv_spk_loss_pros, adv_spk_acc_pros, _ = self.adv_speaker_head_pros.compute_loss(
                self.grl(z_pros), speaker_seq
            )

            adv_ling_loss_spk = self.adv_linguistic_head_spk.compute_loss(
                self.grl(z_spk),
                linguistic_targets,
                input_lengths=lengths,
                target_lengths=target_lengths,
            )

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
            "spk_loss_ling": adv_spk_loss_ling,
            "spk_loss_pros": adv_spk_loss_pros,
            "ling_loss_spk": adv_ling_loss_spk,
            "ling_loss_pros": adv_ling_loss_pros,
        }

        metrics = {
            "spk_acc": spk_output.accuracy,
            "spk_acc_ling": adv_spk_acc_ling,
            "spk_acc_pros": adv_spk_acc_pros,
        }
        states = {"router/layer_probabilities": output.router.layer_probabilities}
        head_outputs = {
            "spk": spk_output,
        }

        loss = RVQDisentanglerLoss(
            rvq=rvq_losses, distill=distill_losses, adv=adv_losses, metrics=metrics, states=states
        )

        return replace(output, loss=loss, head_outputs=head_outputs)

    @torch.inference_mode()
    def inference(
        self,
        features,
        lengths,
        emotion_seq,
        emotion_lengths,
        prosody_seq=None,
    ) -> RVQDisentanglerOutput:
        output = self._shared_step(features, lengths, emotion_seq, emotion_lengths)
        spk_output = self.speaker_head(output.router.zs[0], padding_mask=output.padding_mask)
        return replace(output, head_outputs={"spk": spk_output})

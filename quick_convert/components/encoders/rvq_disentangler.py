from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from quick_convert.components.encoders.conformer_encoder import ConformerEncoderSSL
from quick_convert.components.layers.rvq import RVQOutput
from quick_convert.utils.masking import make_padding_mask, masked_loss, trim_to_min

from .speaker_head import SpeakerASPHead
from .linguistic_head import LinguisticCTCHead
from .linear_head import LinearHead

from ..layers import ResidualVectorQuantizer, GradientReversalLayer, VectorQuantize


@dataclass
class RouterOutput:
    zs: list[torch.Tensor]
    router_loss: torch.Tensor
    layer_mask: Optional[torch.Tensor] = None
    layer_probabilities: Optional[torch.Tensor] = None
    layer_logits: Optional[torch.Tensor] = None


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
            router_loss=router_loss,
            layer_mask=layer_mask,
            layer_probabilities=layer_probabilities,
            layer_logits=layer_logits,
        )

    def compute_loss(self, layer_probabilities: torch.Tensor, one_hot_mask: torch.Tensor) -> torch.Tensor:
        p_expert = layer_probabilities.mean(dim=0)
        f_expert = one_hot_mask.float().mean(dim=0)
        loss = self.n_classes * torch.sum(f_expert * p_expert)
        return loss


@dataclass
class RVQDisentanglerOutput:
    z_q: torch.Tensor
    z_q_list: List[torch.Tensor]

    z_spk: torch.Tensor
    z_ling: torch.Tensor

    commitment_loss: torch.Tensor
    codebook_loss: torch.Tensor
    router_loss: torch.Tensor

    content: torch.Tensor

    z_pros: Optional[torch.Tensor] = None
    padding_mask: Optional[torch.Tensor] = None
    lengths: Optional[torch.Tensor] = None
    layer_mask: Optional[torch.Tensor] = None
    layer_probabilities: Optional[torch.Tensor] = None
    layer_logits: Optional[torch.Tensor] = None
    spk_output: Optional[torch.Tensor] = None


@dataclass
class RVQDisentanglerLossOutput:
    output: RVQDisentanglerOutput
    rvq_losses: dict[str, torch.Tensor]
    distill_losses: dict[str, torch.Tensor]
    adv_losses: Optional[dict[str, torch.Tensor]] = None
    metrics: Optional[dict[str, torch.Tensor]] = None


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
            output = self.encode(features, lengths)
            spk_output = self.speaker_head(output.z_spk)
            return replace(output, spk_output=spk_output)

    def encode(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> RVQDisentanglerOutput:
        content = self.content_encoder(features, padding_mask=padding_mask)

        B, T, F = content.shape
        content = content.transpose(1, 2)

        rvq_output: RVQOutput = self.rvq(content, padding_mask)

        z_q = rvq_output.z_q.transpose(1, 2)
        router_output = self._route(rvq_output.z_qs)

        return RVQDisentanglerOutput(
            z_q=z_q,
            z_q_list=rvq_output.z_qs,
            z_spk=router_output.zs[0],
            z_ling=router_output.zs[1],
            z_pros=router_output.zs[2],
            commitment_loss=rvq_output.commitment_loss,
            codebook_loss=rvq_output.codebook_loss,
            router_loss=router_output.router_loss,
            content=content,
            padding_mask=padding_mask,
            layer_mask=router_output.layer_mask,
            layer_probabilities=router_output.layer_probabilities,
            layer_logits=router_output.layer_logits,
        )

    def _route(self, z_q_list):
        if isinstance(self.router, dict):
            z_spk = z_q_list[self.router["speaker"]].transpose(1, 2)
            z_ling = z_q_list[self.router["linguistic_content"]].transpose(1, 2)

            z_pros = torch.stack(z_q_list[self.router["emo_pros"] :], dim=3).sum(dim=3).transpose(1, 2)
            return RouterOutput(zs=[z_spk, z_ling, z_pros], router_loss=z_q_list[0].new_tensor(0.0))

        router_output = self.router(self.rvq.quantizers, z_q_list, compute_loss=True)
        z_spk = router_output.zs[0].transpose(1, 2)
        z_ling = router_output.zs[1].transpose(1, 2)
        z_pros = router_output.zs[2].transpose(1, 2)

        return replace(router_output, zs=[z_spk, z_ling, z_pros])

    def _shared_step(self, features, lengths, emotion_seq, emotion_lengths):
        features, emotion_seq, lengths = trim_to_min(
            features, emotion_seq, lengths, emotion_lengths, time_dim=1, max_diff=4
        )
        padding_mask = make_padding_mask(lengths, max_length=features.shape[1])
        output = self.encode(features, padding_mask)
        return replace(output, lengths=lengths)

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
    ) -> RVQDisentanglerLossOutput:
        output = self._shared_step(features, lengths, emotion_seq, emotion_lengths)

        rvq_mse_loss = masked_loss(
            F.mse_loss,
            preds=output.z_q,
            targets=output.content.detach().transpose(1, 2),
            mask=output.padding_mask,
        )

        rvq_losses = {
            "commitment_loss": output.commitment_loss,
            "codebook_loss": output.codebook_loss,
            "mse_loss": rvq_mse_loss,
            "load_balancing_loss": output.router_loss,
        }

        spk_output, spk_loss, spk_acc, _ = self.speaker_head.compute_loss(
            output.z_spk, speaker_seq, output.padding_mask
        )

        emo_loss = self.emotion_head.compute_loss(output.z_pros, emotion_seq, output.padding_mask)

        if self.prosody_head is not None and prosody_seq is not None:
            pros_loss = self.prosody_head.compute_loss(output.z_pros, prosody_seq)
        else:
            pros_loss = 0.0

        ctc_loss = self.linguistic_head.compute_loss(
            output.z_ling,
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
            _, adv_spk_loss_ling, adv_spk_acc_ling, _ = self.adv_speaker_head_ling.compute_loss(
                self.grl(output.z_ling), speaker_seq
            )

            _, adv_spk_loss_pros, adv_spk_acc_pros, _ = self.adv_speaker_head_pros.compute_loss(
                self.grl(output.z_pros), speaker_seq
            )

            adv_ling_loss_spk = self.adv_linguistic_head_spk.compute_loss(
                self.grl(output.z_spk),
                linguistic_targets,
                input_lengths=lengths,
                target_lengths=target_lengths,
            )

            adv_ling_loss_pros = self.adv_linguistic_head_pros.compute_loss(
                self.grl(output.z_pros),
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

        metrics = {
            "spk_acc": spk_acc,
            "adv_spk_acc_ling": adv_spk_acc_ling,
            "adv_spk_acc_pros": adv_spk_acc_pros,
        }

        return RVQDisentanglerLossOutput(
            output=replace(output, spk_output=spk_output),
            rvq_losses=rvq_losses,
            distill_losses=distill_losses,
            adv_losses=adv_losses,
            metrics=metrics,
        )

    def inference(
        self,
        features,
        lengths,
        emotion_seq,
        emotion_lengths,
        prosody_seq=None,
    ) -> RVQDisentanglerOutput:
        output = self._shared_step(features, lengths, emotion_seq, emotion_lengths)
        spk_output = self.speaker_head(output.z_spk, padding_mask=output.padding_mask)
        return replace(output, spk_output=spk_output)

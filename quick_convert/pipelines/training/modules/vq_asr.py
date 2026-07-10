from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn

from quick_convert.components.encoders import LinguisticCTCHead
from quick_convert.components.layers import VectorQuantize, LayerWeightedSum
from quick_convert.components.layers.rvq import VQOutput
from quick_convert.components.mixins.resource import OnlineResourceMixin
from quick_convert.components.ssl import ContentEncoder

from torch.optim.lr_scheduler import LRScheduler

from quick_convert.data.types import AudioBatch
from quick_convert.pipelines.training.modules.base import BaseTrainingModule
from quick_convert.utils.masking import make_padding_mask


@dataclass
class VQASRLoss:
    total: torch.Tensor
    ctc_loss: torch.Tensor


@dataclass
class VQASROutput:
    logits: torch.Tensor
    vq: VQOutput
    losses: VQASRLoss

    @property
    def loss(self) -> torch.Tensor:
        return self.losses.total


class VQASRTrainingModule(OnlineResourceMixin, BaseTrainingModule):
    def __init__(
        self,
        quantizer: VectorQuantize,
        ctc_head: LinguisticCTCHead,
        online_encoders: Optional[dict[str, ContentEncoder]] = None,
        optimizer: Callable[..., torch.optim.Optimizer] = torch.optim.AdamW,
        lr_scheduler: Optional[Callable[..., LRScheduler]] = None,
        layer_fusion: Optional[LayerWeightedSum] = None,
        ctc_loss_weight: float = 1.0,
        commitment_loss_weight: float = 1.0,
        codebook_loss_weight: float = 1.0,
    ):
        super().__init__(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

        self.quantizer = quantizer
        self.ctc_head = ctc_head
        self.online_encoders = nn.ModuleDict(online_encoders or {})
        self.layer_fusion = layer_fusion or nn.Identity()
        self.ctc_loss_weight = ctc_loss_weight
        self.commitment_loss_weight = commitment_loss_weight
        self.codebook_loss_weight = codebook_loss_weight

        self.save_hyperparameters(
            ignore=[
                "quantizer",
                "ctc_head",
                "online_encoders",
                "optimizer",
                "lr_scheduler",
            ]
        )
        for encoder in self.online_encoders.values():
            encoder.requires_grad_(False)
            encoder.eval()

    def _shared_step(
        self,
        batch: AudioBatch,
        stage: str,
    ) -> VQASROutput:
        features, feature_lengths = self.get_resource(batch, "content")
        token_ids, token_lengths = self.get_resource(batch, "token_ids")

        padding_mask = make_padding_mask(
            feature_lengths,
            max_length=features.shape[1],
        )

        features = self.layer_fusion(features)
        quantizer_output = self.quantizer(features.transpose(1, 2), padding_mask, loss_reduction="batch_by_sample")

        # back to [B, T, D]
        quantized = quantizer_output.z_q.transpose(1, 2)

        ctc_loss = self.ctc_head.compute_loss(
            quantized,
            token_ids,
            input_lengths=feature_lengths,
            target_lengths=token_lengths,
        )

        loss = (
            self.ctc_loss_weight * ctc_loss
            + self.commitment_loss_weight * quantizer_output.loss.commitment_loss
            + self.codebook_loss_weight * quantizer_output.loss.codebook_loss
        )

        logits = self.ctc_head(quantized)

        self.log_dict(
            {
                f"{stage}/loss": loss,
                f"{stage}/ctc_loss": ctc_loss,
                f"{stage}/vq/commitment_loss": quantizer_output.loss.commitment_loss,
                f"{stage}/vq/codebook_loss": quantizer_output.loss.codebook_loss,
            },
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=len(batch.paths),
        )

        return VQASROutput(
            logits=logits,
            vq=quantizer_output,
            losses=VQASRLoss(total=loss, ctc_loss=ctc_loss),
        )

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Any

import torch
from torch import nn

from quick_convert.components.encoders import LinguisticCTCHead
from quick_convert.components.layers import LayerWeightedSum
from quick_convert.components.layers.heads import HeadOutput, HeadTarget
from quick_convert.components.layers.rvq import BaseResidualVectorQuantizer, RVQOutput
from quick_convert.components.mixins.asr_logging import ASRLoggingMixin
from quick_convert.components.mixins.gradient_logging import ObjectiveGradientLoggingMixin
from quick_convert.components.mixins.resource import OnlineResourceMixin
from quick_convert.components.ssl import ContentEncoder
from quick_convert.data.types import AudioBatch
from quick_convert.utils.masking import make_padding_mask

from ..optim.base import Optimization
from .base import BaseTrainingModule


@dataclass
class VQASROutput:
    vq: RVQOutput
    ctc: HeadOutput
    loss: torch.Tensor


class VQASRTrainingModule(
    ObjectiveGradientLoggingMixin,
    ASRLoggingMixin,
    OnlineResourceMixin,
    BaseTrainingModule,
):
    """Train a single-codebook quantized representation with CTC.

    Configure an EMA residual quantizer with ``n_codebooks=1`` to use the
    exact quantizer behavior and loss contract used by the RVQ systems.
    """

    def __init__(
        self,
        quantizer: BaseResidualVectorQuantizer,
        ctc_head: LinguisticCTCHead,
        optimization: Optimization,
        tokenizer_model_path: PathLike | None = None,
        layer_fusion: LayerWeightedSum | None = None,
        post_quantization_network: nn.Module | None = None,
        online_encoders: dict[str, ContentEncoder] | None = None,
        save_online_encoders: bool = False,
        ctc_loss_weight: float = 1.0,
        use_latents: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            optimization=optimization,
            checkpoint_exclude_prefixes=(() if save_online_encoders else ("online_encoders",)),
        )

        self.quantizer = quantizer
        self.ctc_head = ctc_head
        self.layer_fusion = layer_fusion or nn.Identity()
        self.post_quantization_network = post_quantization_network
        self.online_encoders = nn.ModuleDict(online_encoders or {})
        self.ctc_loss_weight = ctc_loss_weight

        n_codebooks = getattr(quantizer, "n_codebooks", None)
        self.use_latents = use_latents

        if n_codebooks is not None and n_codebooks != 1:
            raise ValueError(
                "VQASRTrainingModule requires a single active codebook; "
                f"configured quantizer has n_codebooks={n_codebooks}."
            )

        self.save_hyperparameters(
            ignore=[
                "quantizer",
                "ctc_head",
                "layer_fusion",
                "post_quantization_network",
            ]
        )

        self.online_encoders.requires_grad_(False)
        self.online_encoders.eval()

        if tokenizer_model_path is not None:
            self.setup_asr_logging(tokenizer_model_path=tokenizer_model_path)

    @property
    def grad_norm_modules(self) -> dict[str, nn.Module]:
        modules: dict[str, nn.Module] = {
            "vq": self.quantizer,
            "ctc_head": self.ctc_head,
        }

        if any(parameter.requires_grad for parameter in self.layer_fusion.parameters()):
            modules["layer_fusion"] = self.layer_fusion

        if self.post_quantization_network is not None:
            modules["post_quantization_network"] = self.post_quantization_network

        return modules

    def _contextualize(
        self,
        quantized: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.post_quantization_network is None:
            return quantized

        return self.post_quantization_network(quantized, padding_mask)

    def forward(self, batch: AudioBatch) -> torch.Tensor:
        content = self.get_resource(batch, "content")
        features = self.layer_fusion(content.values)
        padding_mask = make_padding_mask(content.lengths, max_length=features.shape[1])
        quantizer_output = self.quantizer(features.transpose(1, 2), padding_mask)
        quantized = quantizer_output.z_q.transpose(1, 2)

        return self._contextualize(quantized, padding_mask)

    def _shared_step(
        self,
        batch: AudioBatch,
        stage: str,
    ) -> VQASROutput:
        content = self.get_resource(batch, "content")
        token_ids = self.get_resource(batch, "token_ids")

        features = self.layer_fusion(content.values)
        feature_lengths = content.lengths

        max_feature_length = int(feature_lengths.max().item())
        features = features[:, :max_feature_length]

        padding_mask = make_padding_mask(
            feature_lengths,
            max_length=max_feature_length,
        )

        quantizer_output = self.quantizer(features.transpose(1, 2), padding_mask)
        quantized = quantizer_output.latents if self.use_latents else quantizer_output.z_q

        quantized = quantized.transpose(1, 2)
        contextual_output = self._contextualize(quantized, padding_mask)

        ctc_output = self.ctc_head.compute_loss(
            contextual_output,
            targets=HeadTarget(
                values=token_ids.values,
                lengths=token_ids.lengths,
            ),
            padding_mask=padding_mask,
            lengths=feature_lengths,
        )

        rvq_loss = quantizer_output.loss.loss
        weighted_ctc_loss = self.ctc_loss_weight * ctc_output.loss
        loss = weighted_ctc_loss + rvq_loss

        log_dict = {
            f"{stage}/loss": loss,
            f"{stage}/ctc/loss": ctc_output.loss,
            f"{stage}/ctc/weighted_loss": weighted_ctc_loss,
            f"{stage}/vq/loss": rvq_loss,
        }

        for name, value in quantizer_output.loss.raw.items():
            log_dict[f"{stage}/vq/{name}_loss"] = value

        for name, value in quantizer_output.loss.weighted.items():
            log_dict[f"{stage}/vq/{name}_loss_weighted"] = value

        self.log_dict(
            log_dict,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=(stage == "train"),
            sync_dist=True,
            batch_size=len(batch),
        )

        if stage == "train" and self.global_step % 1000 == 0:
            self._log_codebook_utilization(quantizer_output)
            self._log_objective_gradients(ctc_loss=weighted_ctc_loss, rvq_loss=rvq_loss)

            if isinstance(self.layer_fusion, LayerWeightedSum):
                weights = self.layer_fusion.weights.softmax(dim=-1).squeeze(0)

                self.media_logger.log_bar(
                    "layer_weights",
                    weights,
                    "layer",
                    "weight",
                    step=self.global_step,
                )

        return VQASROutput(vq=quantizer_output, ctc=ctc_output, loss=loss)

    def _log_objective_gradients(
        self,
        ctc_loss: torch.Tensor,
        rvq_loss: torch.Tensor,
    ) -> None:
        objectives = {"ctc": ctc_loss, "vq": rvq_loss}
        parameter_groups = {
            "vq": tuple(parameter for parameter in self.quantizer.parameters() if parameter.requires_grad),
            "layer_fusion": tuple(parameter for parameter in self.layer_fusion.parameters() if parameter.requires_grad),
        }

        if self.post_quantization_network is not None:
            parameter_groups["post_quantization_network"] = tuple(
                parameter for parameter in self.post_quantization_network.parameters() if parameter.requires_grad
            )

        for module_name, parameters in parameter_groups.items():
            if parameters:
                self.log_objective_gradients(
                    losses=objectives,
                    parameters=parameters,
                    prefix=f"objective_grad/{module_name}",
                )

    def _log_codebook_utilization(self, quantizer_output: RVQOutput) -> None:
        codes = quantizer_output.codes
        codebook_size = self.quantizer.codebook_size
        metrics: dict[str, torch.Tensor] = {}

        for layer_idx in range(codes.shape[1]):
            valid_codes = codes[:, layer_idx]
            valid_codes = valid_codes[valid_codes >= 0]
            counts = torch.bincount(valid_codes, minlength=codebook_size).float()
            total = counts.sum().clamp_min(1)
            probabilities = counts / total
            nonzero = probabilities > 0
            entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()
            prefix = f"vq/codebook_{layer_idx}"

            metrics[f"{prefix}/utilization"] = (counts > 0).sum().float() / codebook_size
            metrics[f"{prefix}/perplexity"] = entropy.exp()
            metrics[f"{prefix}/max_usage"] = counts.max() / total

        self.log_dict(metrics, on_step=True, on_epoch=False, sync_dist=True)

    def log_validation_output(
        self,
        batch: AudioBatch,
        output: VQASROutput,
        batch_idx: int,
    ) -> None:
        if self.tokenizer is not None:
            self.log_asr_validation_output(
                batch=batch,
                logits=output.ctc.states["logits"],
                batch_idx=batch_idx,
            )

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
from quick_convert.systems.asr import VQASRSystem

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
            checkpoint_exclude_prefixes=(() if save_online_encoders else ("system.online_encoders",)),
        )

        self.ctc_loss_weight = ctc_loss_weight
        self.system = VQASRSystem(
            quantizer=quantizer,
            ctc_head=ctc_head,
            layer_fusion=layer_fusion,
            post_quantization_network=post_quantization_network,
            online_encoders=online_encoders,
            use_latents=use_latents,
        )

        self.save_hyperparameters(
            ignore=[
                "quantizer",
                "ctc_head",
                "layer_fusion",
                "post_quantization_network",
            ]
        )

        if tokenizer_model_path is not None:
            self.setup_asr_logging(tokenizer_model_path=tokenizer_model_path)

    @property
    def quantizer(self) -> BaseResidualVectorQuantizer:
        return self.system.quantizer

    @property
    def ctc_head(self) -> LinguisticCTCHead:
        return self.system.ctc_head

    @property
    def layer_fusion(self) -> nn.Module:
        return self.system.layer_fusion

    @property
    def post_quantization_network(self) -> nn.Module | None:
        return self.system.post_quantization_network

    @property
    def online_encoders(self) -> nn.ModuleDict:
        return self.system.online_encoders

    @property
    def use_latents(self) -> bool:
        return self.system.use_latents

    def _prepare_checkpoint_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        state_dict = super()._prepare_checkpoint_state_dict(state_dict)
        legacy_prefixes = (
            "quantizer.",
            "ctc_head.",
            "layer_fusion.",
            "post_quantization_network.",
            "online_encoders.",
        )

        for key in list(state_dict):
            if key.startswith(legacy_prefixes):
                state_dict.setdefault(f"system.{key}", state_dict.pop(key))

        return state_dict

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

    def forward(self, batch: AudioBatch) -> torch.Tensor:
        return self.system(batch).contextual

    def _shared_step(
        self,
        batch: AudioBatch,
        stage: str,
    ) -> VQASROutput:
        token_ids = self.get_resource(batch, "token_ids")
        system_output = self.system(batch)

        ctc_output = self.ctc_head.compute_loss_from_logits(
            system_output.logits,
            targets=HeadTarget(
                values=token_ids.values,
                lengths=token_ids.lengths,
            ),
            lengths=system_output.lengths,
        )

        quantizer_output = system_output.quantizer
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

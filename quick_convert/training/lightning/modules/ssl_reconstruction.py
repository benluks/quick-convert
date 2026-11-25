"""Lightning adapter for SSL reconstruction training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from os import PathLike

import torch
from torch import nn

from quick_convert.components.conditioning.cosyvoice_training import training_reference_kwargs
from quick_convert.components.conditioning.paired_audio import paired_audio
from quick_convert.components.generators.acoustic import (
    CosyVoiceDecoderOutput,
    CosyVoiceSpectrogramGenerator,
)
from quick_convert.components.layers import LayerWeightedSum
from quick_convert.components.layers.rvq import BaseResidualVectorQuantizer, RVQOutput
from quick_convert.components.resources.resolution import ResolvedResource
from quick_convert.data import AudioBatch
from quick_convert.systems.conversion import SSLReconstructionConversionSystem
from quick_convert.utils.masking import make_padding_mask

from ..logging.cosyvoice import cosyvoice_validation_media
from ..modules.base import BaseTrainingModule
from ..optim import Optimization


@dataclass
class SSLReconstructionOutput:
    features: torch.Tensor
    lengths: torch.Tensor
    speaker_embedding: torch.Tensor | None
    loss: torch.Tensor
    decoder_output: CosyVoiceDecoderOutput
    encoder_output: RVQOutput | None = None


@dataclass
class SupervisionOutput:
    loss: torch.Tensor


class SSLReconstructionTrainingModule(BaseTrainingModule):
    def __init__(
        self,
        optimization: Optimization,
        system: SSLReconstructionConversionSystem | None = None,
        decoder: CosyVoiceSpectrogramGenerator | None = None,
        feature_transform: nn.Module | None = None,
        *,
        online_encoders: dict[str, nn.Module] | None = None,
        encoder: BaseResidualVectorQuantizer | None = None,
        enable_grad_norm_logging: bool = True,
        reference_seconds: float | None = None,
        content_audio_resource: str | None = None,
        supervision: nn.Module | None = None,
    ):
        super().__init__(optimization=optimization, enable_grad_norm_logging=enable_grad_norm_logging)

        legacy_components = (decoder, feature_transform, online_encoders, encoder)
        if system is not None and any(component is not None for component in legacy_components):
            raise ValueError("Pass either `system` or the legacy SSL reconstruction components, not both.")
        if system is None:
            if decoder is None:
                raise ValueError("SSLReconstructionTrainingModule requires `system` or `decoder`.")
            system = SSLReconstructionConversionSystem(
                decoder=decoder,
                feature_transform=feature_transform,
                online_encoders=online_encoders,
                encoder=encoder,
            )
        self.system = system
        self.supervision = supervision
        if reference_seconds is not None:
            if not math.isfinite(reference_seconds) or reference_seconds <= 0:
                raise ValueError("reference_seconds must be finite and positive.")
            if (
                getattr(self.decoder, "content_encoder", None) is None
                or getattr(self.decoder, "stream_conditioning", None) is None
                or "content" not in self.system.online_encoders
            ):
                raise ValueError(
                    "Separate SSLR references require duration-controlled stream conditioning and an online content encoder."
                )
        self.reference_seconds = reference_seconds
        if content_audio_resource is not None:
            if not content_audio_resource or "content" not in self.system.online_encoders:
                raise ValueError("content_audio_resource requires a resource name and an online content encoder.")
        self.content_audio_resource = content_audio_resource

        self.save_hyperparameters(
            logger=False,
            ignore=[
                "optimization",
                "supervision",
                "system",
                "decoder",
                "feature_transform",
                "online_encoders",
                "encoder",
            ],
        )

    @property
    def encoder(self) -> BaseResidualVectorQuantizer | None:
        return self.system.encoder

    @property
    def decoder(self) -> CosyVoiceSpectrogramGenerator:
        return self.system.decoder

    @property
    def feature_transform(self) -> nn.Module:
        return self.system.feature_transform

    @property
    def online_encoders(self) -> nn.ModuleDict:
        return self.system.online_encoders

    def _prepare_checkpoint_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        state_dict = super()._prepare_checkpoint_state_dict(state_dict)
        legacy_prefixes = (
            "decoder.",
            "feature_transform.",
            "online_encoders.",
            "encoder.",
        )
        for key in list(state_dict):
            if key.startswith(legacy_prefixes):
                state_dict.setdefault(f"system.{key}", state_dict.pop(key))
        return state_dict

    @property
    def grad_norm_modules(self) -> dict[str, nn.Module]:
        if self.supervision is not None:
            return {"decoder": self.decoder, "supervision": self.supervision}
        modules = {f"flow/{name}": module for name, module in self.decoder.flow.named_children()}
        modules["feature_transform"] = self.feature_transform
        if getattr(self.decoder, "content_encoder", None) is not None:
            modules["content_encoder"] = self.decoder.content_encoder
        projection = getattr(self.decoder, "input_projection", None)
        if projection is not None:
            modules["input_projection"] = projection
        if self.encoder is not None:
            modules["encoder"] = self.encoder
        return modules

    def _encode_content(self, content: ResolvedResource) -> tuple[torch.Tensor, torch.Tensor, RVQOutput | None]:
        if content.lengths is None:
            raise ValueError("Content features require valid lengths for SSL reconstruction.")
        return self.system.encode_features(content.values, content.lengths, metadata=content.metadata)

    def _content_resource(self, batch):
        if self.content_audio_resource is not None:
            encoder = self.system.online_encoders["content"]
            batch = paired_audio(batch, self.content_audio_resource, encoder.sample_rate)
        return self.system.get_resource(batch, "content")

    def phase_parameters(self):
        return self.supervision.phase_parameters(self.system)

    def execute_phases(self, batch):
        # Share crop positions across phases, but rebuild the representation graph
        # after optimizer toggling so trainable encoders receive generator gradients.
        return super().execute_phases((batch, torch.rand(len(batch))))

    def phase_step(self, batch, phase):
        batch, positions = batch
        loss = self._supervised_loss(batch, phase, positions=positions)
        self.log(f"train/{phase}/loss", loss, on_step=True, on_epoch=True)
        return SupervisionOutput(loss)

    def _supervised_loss(self, batch, phase, *, positions=None):
        content = self._content_resource(batch)
        features, lengths, encoder_output = self._encode_content(content)
        prepared = self.supervision.prepare_batch(self.system, batch, features, lengths, positions=positions)
        loss = self.supervision.phase_loss(self.system, prepared, phase)
        if phase == "generator" and encoder_output is not None:
            loss = loss + encoder_output.loss.loss
        return loss

    def _shared_step(
        self,
        batch: AudioBatch,
        stage: str,
    ) -> SSLReconstructionOutput:
        if self.supervision is not None:
            loss = self._supervised_loss(batch, "generator")
            self.log(f"{stage}/loss", loss, on_epoch=True, batch_size=len(batch))
            return SupervisionOutput(loss)
        content = self._content_resource(batch)
        speaker_embedding = self.system.speaker_condition(batch)

        features, lengths, encoder_output = self._encode_content(content)

        decoder_output = self.system.decoder.compute_loss(
            features=features,
            lengths=lengths,
            target_wav=batch.waveforms,
            wav_lens=batch.lengths,
            sampling_rate=batch.sample_rates[0],
            speaker_embedding=speaker_embedding,
            **(
                training_reference_kwargs(
                    self.system,
                    batch,
                    None
                    if getattr(self.system.decoder, "content_encoder", None) is not None
                    else self.system.decoder.flow.output_lengths(lengths),
                    reference_seconds=self.reference_seconds,
                )
                if getattr(self.system.decoder, "stream_conditioning", None) is not None
                else {}
            ),
        )

        encoder_loss = features.new_zeros(()) if encoder_output is None else encoder_output.loss.loss
        loss = decoder_output.loss + encoder_loss

        log_dict = {
            f"{stage}/loss": loss,
            f"{stage}/decoder/loss": decoder_output.loss,
        }
        if encoder_output is not None:
            log_dict[f"{stage}/rvq/loss"] = encoder_loss
            for name, value in encoder_output.loss.raw.items():
                log_dict[f"{stage}/rvq/{name}_loss"] = value
            for name, value in encoder_output.loss.weighted.items():
                log_dict[f"{stage}/rvq/{name}_loss_weighted"] = value
        self.log_dict(
            log_dict,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=(stage == "train"),
            sync_dist=True,
            batch_size=len(batch),
        )

        if stage == "train" and self.global_step % 1000 == 0:
            self._log_representation_state(encoder_output, lengths)

        return SSLReconstructionOutput(
            features=features,
            lengths=lengths,
            speaker_embedding=speaker_embedding,
            loss=loss,
            decoder_output=decoder_output,
            encoder_output=encoder_output,
        )

    def _log_representation_state(self, encoder_output, lengths):
        if isinstance(self.feature_transform, LayerWeightedSum) and (
            self._trainer is None or self.trainer.is_global_zero
        ):
            self.media_logger.log_bar(
                "layer_weights",
                self.feature_transform.weights.detach().softmax(-1).squeeze(0),
                "layer",
                "weight",
                step=self.global_step,
            )
        if encoder_output is None:
            return
        codes = encoder_output.codes.detach()
        mask = make_padding_mask(lengths, max_length=codes.shape[-1])
        counts_by_codebook = torch.zeros(
            self.encoder.n_codebooks,
            self.encoder.codebook_size,
            device=codes.device,
        )
        for index in range(codes.shape[1]):
            valid = codes[:, index][mask]
            counts_by_codebook[index] = torch.bincount(valid[valid >= 0], minlength=self.encoder.codebook_size)
        # A fixed-size reduction also works when quantizer dropout activates
        # different numbers of codebooks on different ranks.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(counts_by_codebook)
        metrics = {}
        for index, counts in enumerate(counts_by_codebook):
            total = counts.sum().clamp_min(1)
            probabilities = counts / total
            nonzero = probabilities > 0
            prefix = f"rvq/codebook_{index}"
            metrics[f"{prefix}/utilization"] = (counts > 0).float().mean()
            metrics[f"{prefix}/perplexity"] = (-(probabilities[nonzero] * probabilities[nonzero].log()).sum()).exp() * (
                counts.sum() > 0
            )
            metrics[f"{prefix}/max_usage"] = counts.max() / total
        self.log_dict(metrics, on_step=True, on_epoch=False, sync_dist=False)

    def log_validation_output(self, batch, output, batch_idx):
        if batch_idx != 0:
            return
        if self.supervision is not None:
            return

        media = cosyvoice_validation_media(self.system, batch, output, reference_seconds=self.reference_seconds)
        self.media_logger.log_reconstructed_audio(key="val/reconstruction", media=media, step=self.global_step)

    @torch.inference_mode()
    def inference(self, batch: AudioBatch, run_vocoder: bool = True, to_file: PathLike | None = None):
        self.eval()
        result = self.system(batch, run_vocoder=run_vocoder)
        if to_file is not None:
            self.system.save_generation(result.generation, to_file)
        return result.generation

    @torch.inference_mode()
    def infer_file(
        self,
        path: PathLike,
        run_vocoder: bool = True,
        to_file: PathLike | None = None,
    ):
        return self.system.reconstruct(
            path,
            run_vocoder=run_vocoder,
            to_file=to_file,
        ).generation

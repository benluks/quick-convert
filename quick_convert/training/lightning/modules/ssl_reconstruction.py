"""Lightning adapter for SSL reconstruction training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from os import PathLike

import torch
from torch import nn

from quick_convert.components.decoders import ConditioningStream, CosyVoiceDecoderOutput, CosyVoiceSpectrogramGenerator
from quick_convert.components.layers import LayerWeightedSum
from quick_convert.components.layers.rvq import BaseResidualVectorQuantizer, RVQOutput
from quick_convert.components.mixins.resource import ResolvedResource
from quick_convert.data import AudioBatch, load_paired_audio_batch
from quick_convert.systems.reconstruction import SSLReconstructionSystem
from quick_convert.utils.masking import make_padding_mask

from ..logging.media_logger import ReconstructedAudio
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


class SSLReconstructionTrainingModule(BaseTrainingModule):
    def __init__(
        self,
        optimization: Optimization,
        system: SSLReconstructionSystem | None = None,
        decoder: CosyVoiceSpectrogramGenerator | None = None,
        feature_transform: nn.Module | None = None,
        *,
        online_encoders: dict[str, nn.Module] | None = None,
        encoder: BaseResidualVectorQuantizer | None = None,
        enable_grad_norm_logging: bool = True,
        reference_seconds: float | None = None,
        content_audio_resource: str | None = None,
    ):
        super().__init__(optimization=optimization, enable_grad_norm_logging=enable_grad_norm_logging)

        legacy_components = (decoder, feature_transform, online_encoders, encoder)
        if system is not None and any(component is not None for component in legacy_components):
            raise ValueError("Pass either `system` or the legacy SSL reconstruction components, not both.")
        if system is None:
            if decoder is None:
                raise ValueError("SSLReconstructionTrainingModule requires `system` or `decoder`.")
            system = SSLReconstructionSystem(
                decoder=decoder,
                feature_transform=feature_transform,
                online_encoders=online_encoders,
                encoder=encoder,
            )
        self.system = system
        if reference_seconds is not None:
            if not math.isfinite(reference_seconds) or reference_seconds <= 0:
                raise ValueError("reference_seconds must be finite and positive.")
            if getattr(self.decoder, "content_encoder", None) is None:
                raise ValueError("Separate SSLR references require the duration-controlled decoder.")
        self.reference_seconds = reference_seconds
        if content_audio_resource is not None:
            if not content_audio_resource or "content" not in self.system.online_encoders:
                raise ValueError("content_audio_resource requires a resource name and an online content encoder.")
        self.content_audio_resource = content_audio_resource

        self.save_hyperparameters(
            logger=False,
            ignore=[
                "optimization",
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
        return self.system.encode_features(content.values, content.lengths)

    def _content_resource(self, batch):
        if self.content_audio_resource is not None:
            encoder = self.system.online_encoders["content"]
            batch = load_paired_audio_batch(batch, self.content_audio_resource, encoder.sample_rate)
        return self.system.get_resource(batch, "content")

    def _conditioning_streams(self, batch, target_lengths=None):
        """Extract prefixes from selected references, retaining original targets."""
        if self.reference_seconds is None:
            return self.system.conditioning_streams(batch, target_lengths)
        if batch.sample_rates is None or torch.any(batch.sample_rates != batch.sample_rates[0]):
            raise ValueError("Reference conditioning requires a common source sample rate.")
        sample_rate = int(batch.sample_rates[0])
        max_samples = int(sample_rate * self.reference_seconds)
        if max_samples < 1:
            raise ValueError("reference_seconds must contain at least one sample.")
        reference = load_paired_audio_batch(batch, "reference_path", sample_rate, max_samples)
        streams = self.system.conditioning_streams(reference)
        if "mel" in self.decoder.stream_conditioning.dimensions:
            mels, mel_lengths = self.decoder._compute_mels(reference.waveforms, reference.lengths, sample_rate)
            prefix_lengths = self.decoder.stream_conditioning.sample_prefix_lengths(mel_lengths)
            streams["mel"] = ConditioningStream(
                mels.transpose(1, 2)[:, : int(prefix_lengths.max())], prefix_lengths, is_prefix=True
            )
        return streams

    def _shared_step(
        self,
        batch: AudioBatch,
        stage: str,
    ) -> SSLReconstructionOutput:
        content = self._content_resource(batch)
        speaker_embedding = self.system.speaker_condition(batch)

        features, lengths, encoder_output = self._encode_content(content)

        decoder_output = self.decoder.compute_loss(
            features=features,
            lengths=lengths,
            target_wav=batch.waveforms,
            wav_lens=batch.lengths,
            sampling_rate=batch.sample_rates[0],
            speaker_embedding=speaker_embedding,
            **(
                {
                    "streams": self._conditioning_streams(
                        batch,
                        None
                        if getattr(self.decoder, "content_encoder", None) is not None
                        else self.decoder.flow.output_lengths(lengths),
                    )
                }
                if getattr(self.decoder, "stream_conditioning", None) is not None
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

        stream_kwargs = {}
        if getattr(self.decoder, "stream_conditioning", None) is not None:
            streams = self._conditioning_streams(
                batch,
                None
                if getattr(self.decoder, "content_encoder", None) is not None
                else self.decoder.flow.output_lengths(output.lengths),
            )
            if "mel" in self.decoder.stream_conditioning.dimensions and "mel" not in streams:
                mels, mel_lengths = self.decoder._compute_mels(
                    batch.waveforms, batch.lengths, sampling_rate=batch.sample_rates[0]
                )
                streams["mel"] = ConditioningStream(mels.transpose(1, 2), mel_lengths)
            # Validation exercises prefix-only inference rather than leaking target streams.
            prefixes = {}
            for name, stream in streams.items():
                if stream.is_prefix:
                    prefixes[name] = stream
                    continue
                lengths = (stream.lengths // 5).clamp_min(1)
                prefixes[name] = ConditioningStream(stream.values[:, : int(lengths.max())], lengths)
            stream_kwargs["streams"] = prefixes
        if getattr(self.decoder, "content_encoder", None) is not None:
            target_lengths = self.decoder.mel_output_lengths(batch.lengths, int(batch.sample_rates[0]))
            stream_kwargs["duration_ratio"] = target_lengths / self.decoder.flow.output_lengths(output.lengths)
        generation = self.system.decode_features(
            output.features,
            output.lengths,
            output.speaker_embedding,
            run_vocoder=True,
            **stream_kwargs,
        )
        if generation.audio is None:
            raise RuntimeError("Decoder did not return audio when run_vocoder=True.")
        original_mel, original_mel_lengths = self.decoder._compute_mels(
            batch.waveforms, batch.lengths, sampling_rate=batch.sample_rates[0]
        )
        self.media_logger.log_reconstructed_audio(
            key="val/reconstruction",
            media=ReconstructedAudio(
                original_audio=batch.waveforms,
                reconstructed_audio=generation.audio.waveforms,
                original_mel=original_mel,
                reconstructed_mel=generation.mel,
                original_audio_lengths=batch.lengths,
                reconstructed_audio_lengths=generation.audio.lengths,
                original_mel_lengths=original_mel_lengths,
                reconstructed_mel_lengths=generation.mel_lengths,
                original_sample_rate=batch.sample_rates[0],
                reconstructed_sample_rate=generation.audio.sample_rate,
                ids=batch.utt_ids,
            ),
            step=self.global_step,
        )

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

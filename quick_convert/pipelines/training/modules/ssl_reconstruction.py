from __future__ import annotations

from dataclasses import dataclass
from os import PathLike

import torch
from torch import nn

from quick_convert.components.decoders import CosyVoiceDecoderOutput, CosyVoiceSpectrogramGenerator
from quick_convert.components.layers.rvq import BaseResidualVectorQuantizer, RVQOutput
from quick_convert.components.mixins.resource import ResolvedResource
from quick_convert.data.types import AudioBatch
from quick_convert.systems.reconstruction import SSLReconstructionSystem

from ..logging.media_logger import ReconstructedAudio
from ..modules.base import BaseTrainingModule
from ..optim.base import Optimization


@dataclass
class SSLReconstructionOutput:
    features: torch.Tensor
    lengths: torch.Tensor
    speaker_embedding: torch.Tensor
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
    ):
        super().__init__(optimization=optimization)

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

        self.save_hyperparameters(
            ignore=[
                "system",
                "decoder",
                "feature_transform",
                "online_encoders",
                "encoder",
            ]
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
        if self.encoder is not None:
            modules["encoder"] = self.encoder
        return modules

    def _encode_content(self, content: ResolvedResource) -> tuple[torch.Tensor, torch.Tensor, RVQOutput | None]:
        if content.lengths is None:
            raise ValueError("Content features require valid lengths for SSL reconstruction.")
        return self.system.encode_features(content.values, content.lengths)

    def _shared_step(
        self,
        batch: AudioBatch,
        stage: str,
    ) -> SSLReconstructionOutput:
        content = self.system.get_resource(batch, "content")
        speaker_embedding = self.system.get_resource(batch, "speaker").values

        features, lengths, encoder_output = self._encode_content(content)

        decoder_output = self.decoder.compute_loss(
            features=features,
            lengths=lengths,
            target_wav=batch.waveforms,
            wav_lens=batch.lengths,
            sampling_rate=batch.sample_rates[0],
            speaker_embedding=speaker_embedding,
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

        return SSLReconstructionOutput(
            features=features,
            lengths=lengths,
            speaker_embedding=speaker_embedding,
            loss=loss,
            decoder_output=decoder_output,
            encoder_output=encoder_output,
        )

    def log_validation_output(self, batch, output, batch_idx):
        if batch_idx != 0:
            return

        generation = self.system.decode_features(
            output.features,
            output.lengths,
            output.speaker_embedding,
            run_vocoder=True,
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

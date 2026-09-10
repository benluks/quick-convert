from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import torch
import torchaudio
from torch import nn

from quick_convert.components.decoders import CosyVoiceDecoderOutput, CosyVoiceSpectrogramGenerator
from quick_convert.components.layers.rvq import BaseResidualVectorQuantizer, RVQOutput
from quick_convert.components.mixins.resource import OnlineResourceMixin
from quick_convert.components.ssl.base import ContentEncoder
from quick_convert.data.types import AudioBatch
from quick_convert.utils.audio import load_audio
from quick_convert.utils.masking import make_padding_mask

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


class SSLReconstructionTrainingModule(OnlineResourceMixin, BaseTrainingModule):
    def __init__(
        self,
        decoder: CosyVoiceSpectrogramGenerator,
        feature_transform: nn.Module | None,
        optimization: Optimization,
        *,
        online_encoders: dict[str, ContentEncoder] | None = None,
        encoder: BaseResidualVectorQuantizer | None = None,
    ):
        super().__init__(optimization=optimization)

        self.encoder = encoder
        self.decoder = decoder
        self.feature_transform = feature_transform or nn.Identity()
        self.online_encoders = nn.ModuleDict(online_encoders or {})

        self.online_encoders.requires_grad_(False)
        self.online_encoders.eval()
        # for encoder in self.online_encoders.values():
        #     encoder.requires_grad_(False)
        #     encoder.eval()

    @property
    def grad_norm_modules(self) -> dict[str, nn.Module]:
        modules = {f"flow/{name}": module for name, module in self.decoder.flow.named_children()}
        if self.encoder is not None:
            modules["encoder"] = self.encoder
        return modules

    def _encode_content(self, content) -> tuple[torch.Tensor, torch.Tensor, RVQOutput | None]:
        if content.lengths is None:
            raise ValueError("Content features require valid lengths for SSL reconstruction.")

        features = self.feature_transform(content.values)
        lengths = content.lengths

        if hasattr(self.feature_transform, "output_lengths"):
            lengths = self.feature_transform.output_lengths(lengths)
        elif not isinstance(self.feature_transform, nn.Identity):
            raise TypeError("SSL reconstruction feature_transform modules must expose output_lengths(input_lengths).")

        if lengths.shape != (features.shape[0],):
            raise ValueError(
                f"Expected one content length per batch item, got {tuple(lengths.shape)} "
                f"for batch size {features.shape[0]}."
            )
        if torch.any(lengths > features.shape[1]):
            raise ValueError(
                "A transformed content length exceeds the padded feature time dimension: "
                f"maximum is {features.shape[1]}, got {int(lengths.max())}."
            )

        if self.encoder is None:
            return features, lengths, None

        padding_mask = make_padding_mask(lengths, max_length=features.shape[1])
        encoder_output = self.encoder(features.transpose(1, 2), padding_mask)
        encoded_lengths = self.encoder.output_lengths(lengths)

        if torch.any(encoded_lengths > encoder_output.z_q.shape[2]):
            raise RuntimeError(
                "An encoder output length exceeds the quantized time dimension: "
                f"maximum is {encoder_output.z_q.shape[2]}, got {int(encoded_lengths.max())}."
            )

        return encoder_output.z_q.transpose(1, 2), encoded_lengths, encoder_output

    def _shared_step(
        self,
        batch: AudioBatch,
        stage: str,
    ) -> SSLReconstructionOutput:
        content = self.get_resource(batch, "content")
        speaker_embedding = self.get_resource(batch, "speaker").values

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

        generation = self.decoder(
            feature=output.features,
            length=output.lengths,
            speaker_embedding=output.speaker_embedding,
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

        content = self.get_resource(batch, "content")
        speaker_embedding = self.get_resource(batch, "speaker")

        features, lengths, _ = self._encode_content(content)

        generation = self.decoder(
            feature=features,
            length=lengths,
            speaker_embedding=speaker_embedding.values,
            run_vocoder=run_vocoder,
        )

        if to_file is not None:
            if generation.audio is None:
                raise ValueError("to_file requires run_vocoder=True.")
            if len(generation.audio) != 1:
                raise ValueError("to_file only supports a single generated waveform.")
            torchaudio.save(
                Path(to_file),
                generation.audio.waveform(0).unsqueeze(0).to("cpu"),
                generation.audio.sample_rate,
            )

        return generation

    @torch.inference_mode()
    def infer_file(self, path: PathLike, *args, **kwargs):
        waveform, sample_rate = load_audio(path)

        waveform = waveform.to(self.device)

        batch = AudioBatch(
            utt_ids=[Path(path).stem],
            paths=[Path(path)],
            splits=[""],
            waveforms=waveform.reshape(1, -1),
            lengths=torch.tensor(
                [waveform.shape[-1]],
                device=self.device,
            ),
            sample_rates=torch.tensor([sample_rate], device=self.device),
            resources={},
        )

        return self.inference(batch, *args, **kwargs)

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import torch
import torchaudio
from torch import nn

from quick_convert.components.decoders.chatterbox import ChatterboxDecoderOutput, ChatterboxSpectrogramGenerator
from quick_convert.components.mixins.resource import OnlineResourceMixin
from quick_convert.components.ssl.base import ContentEncoder
from quick_convert.data.types import AudioBatch
from quick_convert.pipelines.training.logging.base import ReconstructedAudio
from quick_convert.utils.audio import load_audio

from ..modules.base import BaseTrainingModule, TrainingStepOutput
from ..optim.base import Optimization


@dataclass
class SSLReconstructionOutput:
    features: float["b t d"]
    lengths: float["b"]
    speaker_embedding: float["b d_spk"]
    loss: float
    decoder_output: ChatterboxDecoderOutput


class SSLReconstructionTrainingModule(
    OnlineResourceMixin,
    BaseTrainingModule,
):
    def __init__(
        self,
        decoder: ChatterboxSpectrogramGenerator,
        feature_transform: nn.Module,
        optimization: Optimization,
        *,
        online_encoders: dict[str, ContentEncoder] | None = None,
    ):
        super().__init__(optimization=optimization)

        self.decoder = decoder
        self.feature_transform = feature_transform
        self.online_encoders = nn.ModuleDict(online_encoders or {})

        self.online_encoders.requires_grad_(False)
        self.online_encoders.eval()
        # for encoder in self.online_encoders.values():
        #     encoder.requires_grad_(False)
        #     encoder.eval()
        self.grad_norm_modules = {
            "flow.encoder": self.decoder.flow.encoder,
            "flow.decoder.estimator": self.decoder.flow.decoder.estimator,
            "flow.decoder": self.decoder.flow.decoder,
        }

    def _shared_step(
        self,
        batch: AudioBatch,
        stage: str,
    ) -> TrainingStepOutput:
        features = self.get_resource(
            batch,
            "content",
        )
        speaker_embedding = self.get_resource(batch, "speaker").values

        lengths = features.lengths
        features = self.feature_transform(features.values)

        decoder_output = self.decoder.compute_loss(
            features=features,
            lengths=lengths,
            target_wav=batch.waveforms,
            wav_lens=batch.lengths,
            sampling_rate=batch.sample_rates[0],
            speaker_embedding=speaker_embedding,
        )

        log_dict = {
            f"{stage}/loss": decoder_output.loss,
        }

        self.log_dict(
            log_dict,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=(stage == "train"),
            sync_dist=True,
            batch_size=len(batch),
        )
        weights = self.feature_transform.weights.softmax(dim=-1).squeeze(0)

        self.media_logger.log_bar(
            "layer_weights",
            weights,
            step=self.global_step,
        )

        return SSLReconstructionOutput(
            features=features,
            lengths=lengths,
            speaker_embedding=speaker_embedding,
            loss=decoder_output.loss,
            decoder_output=decoder_output,
        )

    def log_validation_output(self, batch, output, batch_idx):
        reconstructed_mel, reconstructed_wav = self.decoder(
            feature=output.features,
            length=output.lengths,
            speaker_embedding=output.speaker_embedding,
            run_vocoder=True,
        )
        original_mel, original_mel_lengths = self.decoder._compute_mels(
            batch.waveforms, batch.lengths, sampling_rate=batch.sample_rates[0]
        )
        self.media_logger.log_reconstructed_audio(
            key="val/reconstruction",
            media=ReconstructedAudio(
                original_audio=batch.waveforms,
                reconstructed_audio=reconstructed_wav,
                original_mel=original_mel,
                reconstructed_mel=reconstructed_mel,
                audio_lengths=batch.lengths,
                mel_lengths=original_mel_lengths,
                original_sample_rate=batch.sample_rates[0],
                reconstructed_sample_rate=self.decoder.VOCODER_SR,
                ids=batch.utt_ids,
            ),
            step=self.global_step,
        )

    @torch.inference_mode()
    def inference(self, batch: AudioBatch, run_vocoder: True, to_file: PathLike | None = None) -> torch.Tensor:
        self.eval()

        content = self.get_resource(batch, "content")
        speaker_embedding = self.get_resource(batch, "speaker")

        features = self.feature_transform(content.values)

        mel, wav = self.decoder(
            feature=features,
            length=content.lengths,
            speaker_embedding=speaker_embedding.values,
            run_vocoder=run_vocoder,
        )

        if run_vocoder and to_file is not None:
            torchaudio.save(Path(to_file), wav.to("cpu"), self.decoder.VOCODER_SR)

        return mel, wav

    @torch.inference_mode()
    def infer_file(self, path: PathLike, *args, **kwargs) -> torch.Tensor:
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

    # def log_validation_output(self, batch, output, batch_idx):
    #     return self.logger.experiment.add_audio()

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Literal

import torch
import torchaudio
from torch import nn

from quick_convert.components.decoders import CosyVoiceDecoderOutput, CosyVoiceSpectrogramGenerator
from quick_convert.components.encoders.rvq_disentangler import RVQDisentangler, RVQDisentanglerOutput
from quick_convert.components.layers.routers import LearnedRVQLayerRouter
from quick_convert.components.mixins.gradient_logging import ObjectiveGradientLoggingMixin
from quick_convert.components.mixins.resource import OnlineResourceMixin
from quick_convert.components.ssl.base import ContentEncoder
from quick_convert.data.types import AudioBatch
from quick_convert.utils.audio import load_audio

from ..logging.media_logger import ReconstructedAudio
from ..modules.base import BaseTrainingModule
from ..optim.base import Optimization


@dataclass
class SSLReconstructionOutput:
    features: float["b t d"]
    lengths: float["b"]
    speaker_embedding: float["b d_spk"]
    loss: float
    decoder_output: CosyVoiceDecoderOutput


class SSLReconstructionTrainingModule(
    ObjectiveGradientLoggingMixin,
    OnlineResourceMixin,
    BaseTrainingModule,
):
    def __init__(
        self,
        decoder: CosyVoiceSpectrogramGenerator,
        feature_transform: nn.Module | None,
        optimization: Optimization,
        *,
        online_encoders: dict[str, ContentEncoder] | None = None,
        encoder: RVQDisentangler | None = None,
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
        modules: dict[str, nn.Module] = {f"flow/{name}": module for name, module in self.decoder.flow.named_children()}

        if self.encoder is not None:
            modules["encoder/content_encoder"] = self.encoder.content_encoder
            modules["encoder/rvq"] = self.encoder.rvq

            if isinstance(
                self.encoder.router,
                LearnedRVQLayerRouter,
            ):
                modules["encoder/router"] = self.encoder.router

        return modules

    def _shared_step(
        self,
        batch: AudioBatch,
        stage: str,
    ) -> SSLReconstructionOutput:
        content = self.get_resource(batch, "content")
        speaker_embedding = self.get_resource(batch, "speaker").values

        features = self.feature_transform(content.values)
        lengths = content.lengths

        if self.encoder is None:
            rvq_loss = features.new_zeros(())
            encoder_output = None
        else:
            encoder_output = self.encoder.compute_loss(
                features=features,
                lengths=lengths,
                head_targets={},
                run_adv=False,
            )

            features = encoder_output.rvq.z_q
            lengths = encoder_output.lengths

            # losses from encoder output
            rvq_losses = encoder_output.loss.rvq
            rvq_loss = rvq_losses.loss
            latent_reconstruction_loss = encoder_output.loss.latent_reconstruction
            router_loss = encoder_output.loss.load_balancing

        decoder_output = self.decoder.compute_loss(
            features=features,
            lengths=lengths,
            target_wav=batch.waveforms,
            wav_lens=batch.lengths,
            sampling_rate=batch.sample_rates[0],
            speaker_embedding=speaker_embedding,
        )

        loss = decoder_output.loss + rvq_loss + latent_reconstruction_loss + router_loss

        log_dict = {
            f"{stage}/loss": loss,
            f"{stage}/decoder/loss": decoder_output.loss,
        }
        if encoder_output is not None:
            log_dict[f"{stage}/rvq/loss"] = rvq_loss
            log_dict[f"{stage}/rvq/latent_reconstruction_weighted_loss"] = latent_reconstruction_loss
            log_dict[f"{stage}/rvq/router_loss"] = router_loss

            for name, value in rvq_losses.raw.items():
                log_dict[f"{stage}/rvq/{name}_loss"] = value

            for name, value in rvq_losses.weighted.items():
                log_dict[f"{stage}/rvq/{name}_loss_weighted"] = value

            self._log_encoder_state(
                encoder_output=encoder_output,
                stage=stage,
                decoder_loss=decoder_output.loss,
                rvq_loss=rvq_loss,
            )

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
        )

    def _log_encoder_state(
        self,
        encoder_output: RVQDisentanglerOutput,
        decoder_loss,
        rvq_loss,
        stage: Literal["train", "valid", "test"],
    ) -> None:
        if stage != "train" or self.global_step % 1000 != 0:
            return

        self._log_codebook_utilization(encoder_output)

        self._log_encoder_objective_gradients(
            decoder_loss=decoder_loss,
            rvq_loss=rvq_loss,
        )

        if self.encoder.layer_transform is not None:
            weights = self.encoder.layer_transform.weights.softmax(dim=-1).squeeze(0)

            self.media_logger.log_bar(
                "layer_weights",
                weights,
                "layer",
                "weight",
                step=self.global_step,
            )

        if isinstance(self.encoder.router, LearnedRVQLayerRouter):
            probs = encoder_output.loss.states["router_probabilities"]
            logits = encoder_output.loss.states["router_logits"]

            self._log.log_heatmap(
                "router/probabilities",
                probs,
                step=self.global_step,
                x_labels=list(encoder_output.router.zs.keys()),
                y_labels=[f"RVQ {i}" for i in range(probs.shape[0])],
                annotate=True,
                vmin=0.0,
                vmax=1.0,
            )

            self._log.log_heatmap(
                "router/logits",
                logits,
                step=self.global_step,
                x_labels=list(encoder_output.router.zs.keys()),
                y_labels=[f"RVQ {i}" for i in range(logits.shape[0])],
                annotate=True,
            )

    def _log_encoder_objective_gradients(
        self,
        decoder_loss: torch.Tensor,
        rvq_loss: torch.Tensor,
    ) -> None:
        if self.encoder is None:
            return

        objectives = {
            "decoder": decoder_loss,
            "rvq": rvq_loss,
        }

        parameter_groups = {
            "content_encoder": tuple(p for p in self.encoder.content_encoder.parameters() if p.requires_grad),
            "rvq": tuple(p for p in self.encoder.rvq.parameters() if p.requires_grad),
        }

        for module_name, parameters in parameter_groups.items():
            self.log_objective_gradients(
                losses=objectives,
                parameters=parameters,
                prefix=f"objective_grad/{module_name}",
            )

        if isinstance(self.encoder.router, LearnedRVQLayerRouter):
            router_parameters = tuple(p for p in self.encoder.router.parameters() if p.requires_grad)

            self.log_objective_gradients(
                losses={"rvq": rvq_loss},
                parameters=router_parameters,
                prefix="objective_grad/router",
            )

    def log_validation_output(self, batch, output, batch_idx):
        if batch_idx != 0:
            return

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

    def _log_codebook_utilization(
        self,
        encoder_output: RVQDisentanglerOutput,
    ) -> None:
        codes = encoder_output.rvq.codes
        padding_mask = encoder_output.padding_mask.squeeze(-1).bool()

        codebook_size = self.encoder.rvq.codebook_size

        metrics: dict[str, torch.Tensor] = {}

        for layer_idx in range(codes.shape[1]):
            layer_codes = codes[:, layer_idx]
            valid_codes = layer_codes[padding_mask]

            counts = torch.bincount(
                valid_codes,
                minlength=codebook_size,
            ).float()

            total = counts.sum().clamp_min(1)

            utilization = (counts > 0).sum().float() / codebook_size

            probabilities = counts / total
            nonzero = probabilities > 0

            entropy = -(probabilities[nonzero] * probabilities[nonzero].log()).sum()

            perplexity = entropy.exp()
            max_usage = counts.max() / total

            prefix = f"rvq/codebook_{layer_idx}"

            metrics[f"{prefix}/utilization"] = utilization
            metrics[f"{prefix}/perplexity"] = perplexity
            metrics[f"{prefix}/max_usage"] = max_usage

        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )

    @torch.inference_mode()
    def inference(
        self, batch: AudioBatch, run_vocoder: bool = True, to_file: PathLike | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self.eval()

        content = self.get_resource(batch, "content")
        speaker_embedding = self.get_resource(batch, "speaker")

        features = self.feature_transform(content.values)
        lengths = content.lengths

        if self.encoder is not None:
            encoder_output = self.encoder(features, lengths)
            features = encoder_output.rvq.z_q
            lengths = encoder_output.lengths

        mel, wav = self.decoder(
            feature=features,
            length=lengths,
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

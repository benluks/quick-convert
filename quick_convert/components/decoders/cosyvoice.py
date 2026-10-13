from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F
from torch import nn

from quick_convert.components.feature_extractors.mel import MelSpectrogramConfig
from quick_convert.data import GeneratedAudio
from quick_convert.utils.device import DeviceLike, configure_device
from quick_convert.utils.masking import make_padding_mask, trim_to_min


if TYPE_CHECKING:
    from ...external.cosyvoice.flow.flow import CausalMaskedDiffWithDiT, CausalMaskedDiffWithXvec
    from .hift_generator import CosyVoiceHiFTDecoder


@dataclass
class CosyVoiceDecoderOutput:
    flow_state: torch.Tensor | None = None
    loss: torch.Tensor | None = None


@dataclass(frozen=True)
class CosyVoiceGenerationOutput:
    """Generated mel features and optional waveform output with valid lengths."""

    mel: torch.Tensor
    mel_lengths: torch.LongTensor
    audio: GeneratedAudio | None = None


class CosyVoiceSpectrogramGenerator(nn.Module):
    def __init__(
        self,
        flow: CausalMaskedDiffWithXvec | CausalMaskedDiffWithDiT,
        # feature dim output by preceding encoder.
        # if encoder outputs discrete tokens, set to `None` and ensure
        # `flow` has `vocab_size` set to an integer value
        feature_dim: int | None,
        cond_strategy: Literal["rvq", "mel"] | None = None,
        device: DeviceLike = None,
        mel_dim: int = 80,
        vocoder: CosyVoiceHiFTDecoder | None = None,
        vocoder_repo_id: str = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        vocoder_filename: str = "hift.pt",
        vocoder_config_filename: str = "cosyvoice3.yaml",
        vocoder_revision: str | None = None,
        mel_config: MelSpectrogramConfig | None = None,
    ):
        super().__init__()

        self.device = configure_device(device)
        self.flow = flow
        self.cond_strategy = cond_strategy
        self.vocoder = vocoder
        self.vocoder_repo_id = vocoder_repo_id
        self.vocoder_filename = vocoder_filename
        self.vocoder_config_filename = vocoder_config_filename
        self.vocoder_revision = vocoder_revision
        self.mel_dim = mel_dim
        if mel_dim != flow.output_size:
            raise ValueError("mel_dim must match the flow output size.")
        self.register_buffer("_mel_metadata", torch.empty(0, dtype=torch.uint8))
        self._resolved_mel_metadata = None
        if mel_config is not None:
            self._set_mel_metadata({"mel_config": asdict(mel_config)})
        self.input_projection = None
        if flow.vocab_size is None and feature_dim != flow.input_size:
            self.input_projection = nn.Linear(feature_dim, flow.input_size, device=self.device)

    def initialize_checkpoint_vocoder(self, config: dict | None = None) -> None:
        """Construct HiFT for strict checkpoint restoration without downloading weights."""
        from .hift_generator import CosyVoiceHiFTConfig, CosyVoiceHiFTDecoder, F0PredictorConfig

        config = config or {}
        self.vocoder = (
            CosyVoiceHiFTDecoder(
                config=CosyVoiceHiFTConfig(**config.get("config", {})),
                f0_config=F0PredictorConfig(**config.get("f0_config", {})),
                device=self.device,
                mel_config=MelSpectrogramConfig(**config["mel_config"]) if config.get("mel_config") else None,
            )
            .eval()
            .requires_grad_(False)
        )

    def _get_vocoder(self) -> CosyVoiceHiFTDecoder:
        if self.vocoder is None:
            from .hift_generator import CosyVoiceHiFTDecoder

            metadata = self._metadata()
            if "config" not in metadata:
                pretrained = CosyVoiceHiFTDecoder.pretrained_config(
                    self.vocoder_repo_id,
                    self.vocoder_config_filename,
                    self.vocoder_revision,
                )
                if MelSpectrogramConfig(**pretrained["mel_config"]) != self.mel_config:
                    raise ValueError("Configured mel specification differs from the pretrained vocoder specification.")
                pretrained["source"] = {"repo_id": self.vocoder_repo_id, "filename": self.vocoder_filename}
                self._set_mel_metadata(pretrained)
                metadata = pretrained
            source = metadata.get("source", {})
            vocoder = CosyVoiceHiFTDecoder.from_pretrained(
                repo_id=source.get("repo_id", self.vocoder_repo_id),
                filename=source.get("filename", self.vocoder_filename),
                config_filename=self.vocoder_config_filename,
                revision=self.vocoder_revision,
                metadata=metadata,
                device=self.device,
            )
            if vocoder.mel_config != self.mel_config:
                raise ValueError("Configured mel specification differs from the pretrained vocoder specification.")
            self.vocoder = vocoder
        return self.vocoder

    def mel_output_lengths(self, sample_lengths: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        """Return frame counts after resampling to the resolved mel specification."""
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive.")
        spec = self.mel_config
        resampled = torch.div(
            sample_lengths * spec.sample_rate + sampling_rate - 1,
            sampling_rate,
            rounding_mode="floor",
        )
        return spec.output_lengths(resampled)

    def _set_mel_metadata(self, metadata: dict) -> None:
        spec = MelSpectrogramConfig(**metadata["mel_config"])
        if spec.num_mels != self.mel_dim:
            raise ValueError("Mel specification must match the flow output size.")
        if self.vocoder is not None:
            vocoder_spec = getattr(self.vocoder, "mel_config", None)
            if vocoder_spec is not None and vocoder_spec != spec:
                raise ValueError("Mel specification differs from the custom vocoder specification.")
            if (spec.sample_rate, spec.hop_size) != (self.vocoder.sample_rate, self.vocoder.samples_per_frame):
                raise ValueError("Mel specification must match the vocoder rate and hop.")
        self._resolved_mel_metadata = metadata
        self._mel_metadata = torch.tensor(
            list(json.dumps(metadata).encode()),
            dtype=torch.uint8,
            device=self._mel_metadata.device,
        )

    def _metadata(self) -> dict:
        if self._resolved_mel_metadata is None:
            if self._mel_metadata.numel():
                self._set_mel_metadata(json.loads(bytes(self._mel_metadata.cpu().tolist())))
            elif self.vocoder is not None:
                spec = getattr(self.vocoder, "mel_config", None)
                if spec is None:
                    raise ValueError("A custom vocoder requires an explicit mel_config.")
                self._set_mel_metadata({"mel_config": asdict(spec)})
            else:
                from .hift_generator import CosyVoiceHiFTDecoder

                metadata = CosyVoiceHiFTDecoder.pretrained_config(
                    self.vocoder_repo_id,
                    self.vocoder_config_filename,
                    self.vocoder_revision,
                )
                metadata["source"] = {"repo_id": self.vocoder_repo_id, "filename": self.vocoder_filename}
                self._set_mel_metadata(metadata)
        return self._resolved_mel_metadata

    @property
    def mel_config(self) -> MelSpectrogramConfig:
        """Resolved input feature contract, loaded independently of vocoder weights."""
        return MelSpectrogramConfig(**self._metadata()["mel_config"])

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        metadata = self._metadata()
        if self.vocoder is not None and "config" not in metadata:
            checkpoint_config = getattr(self.vocoder, "checkpoint_config", None)
            if callable(checkpoint_config):
                self._set_mel_metadata({**checkpoint_config(), **metadata})
        super()._save_to_state_dict(destination, prefix, keep_vars)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + "_mel_metadata"
        if key in state_dict:
            saved = state_dict[key]
            if saved.numel():
                metadata = json.loads(bytes(saved.cpu().tolist()))
                if self.vocoder is None and any(name.startswith(prefix + "vocoder.") for name in state_dict):
                    if "config" not in metadata:
                        raise ValueError("Saved vocoder weights require architecture metadata or a composed vocoder.")
                    self.initialize_checkpoint_vocoder(metadata)
                if self._resolved_mel_metadata is not None and self.mel_config != MelSpectrogramConfig(
                    **metadata["mel_config"]
                ):
                    raise ValueError("Checkpoint mel specification differs from the configured specification.")
                self._set_mel_metadata(metadata)
            self._mel_metadata = torch.empty_like(saved, device=self._mel_metadata.device)
            self._resolved_mel_metadata = None
        else:
            raise ValueError("Checkpoint has no mel specification; legacy mel targets require explicit migration.")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _compute_mels(self, wav: torch.Tensor, lengths: torch.Tensor, sampling_rate: int, max_len=None):
        mel, mel_lengths = self.mel_config.extract(wav, lengths, int(sampling_rate))
        if max_len is not None and max_len > mel.shape[-1]:
            mel = F.pad(mel, (0, max_len - mel.shape[-1]))
        return mel, mel_lengths

    def mel2wav(self, mel: torch.Tensor) -> torch.Tensor:
        """Convert mel spectrograms to waveforms, loading the default vocoder on first use."""
        return self._get_vocoder()(mel)

    def compute_loss(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        target_wav: torch.Tensor,
        wav_lens: torch.Tensor,
        sampling_rate: int,
        speaker_embedding: torch.Tensor,
        mask: torch.Tensor = None,
        # cond: Optional[torch.Tensor] = None,
    ):
        """
        Thin wrapper around donor compute_loss.
        """

        input_rate = getattr(self.flow, "input_frame_rate", None)
        ratio = getattr(self.flow, "token_mel_ratio", None)
        if input_rate is not None and ratio is not None:
            if abs(input_rate * ratio - self.mel_config.sample_rate / self.mel_config.hop_size) > 1e-6:
                raise ValueError("Flow output frame rate differs from the mel specification.")
        target_mel, target_mel_lengths = self._compute_mels(
            target_wav, wav_lens, int(sampling_rate), max_len=features.shape[1]
        )

        # W2V-BERT and Matcha both produce 50 Hz features, but their exact
        # boundary conventions differ: W2V-BERT does not pad waveform edges,
        # while Matcha reflect-pads them. A one-frame difference is therefore
        # legitimate and is reconciled explicitly at this training boundary.
        features, target_mel, lengths = trim_to_min(
            features.transpose(1, 2),
            target_mel,
            lengths,
            target_mel_lengths,
        )

        if self.input_projection is not None:
            mask = make_padding_mask(lengths, max_length=features.shape[-1])
            features = self.input_projection(features.transpose(1, 2)) * mask.unsqueeze(-1)
        else:
            features = features.transpose(1, 2)

        batch = {
            "speech_token": features,
            "speech_token_len": lengths,
            "speech_feat": target_mel.transpose(1, 2),
            # use legnths output from the trim function above
            "speech_feat_len": lengths,
            "embedding": speaker_embedding,
        }

        output = self.flow(
            batch=batch,
            # mask=mask,
            device=target_mel.device,
            # cond_strategy=self.cond_strategy,
        )

        return CosyVoiceDecoderOutput(loss=output["loss"])

    @torch.inference_mode()
    def forward(
        self,
        feature: torch.Tensor,
        length: torch.Tensor,
        speaker_embedding: torch.Tensor,
        run_vocoder: bool = False,
    ) -> CosyVoiceGenerationOutput:

        length = length.clone()

        # because of different feature extractions, sometimes lengths can be
        # 1 frame shorter than the features' actual shape, so we trim to the
        # smaller of the 2 values
        max_length_idx = length.argmax()
        min_max_length = min(feature.shape[1], length[max_length_idx])

        feature = feature[:, :min_max_length]
        length[max_length_idx] = min_max_length

        if self.input_projection is not None:
            mask = make_padding_mask(length, max_length=feature.shape[1])
            feature = self.input_projection(feature) * mask.unsqueeze(-1)

        B, _, D = feature.shape
        mel, _ = self.flow.inference(
            token=feature,
            token_len=length,
            prompt_token=torch.zeros(B, 0, D).long().to(self.device),
            prompt_token_len=torch.zeros((B,)).long().to(self.device),
            prompt_feat=torch.zeros(B, 0, self.flow.output_size).to(self.device),
            prompt_feat_len=torch.zeros((B,)).long().to(self.device),
            embedding=speaker_embedding,
            finalize=True,
            streaming=False,
            # max_feature_len=max_len,
            # n_timesteps=n_timesteps,
        )

        mel_lengths = self.flow.output_lengths(length)
        if torch.any(mel_lengths > mel.shape[-1]):
            raise RuntimeError("A generated mel length exceeds the padded mel tensor.")

        if run_vocoder:
            vocoder = self._get_vocoder()
            wav = vocoder(mel)  # mel must be `B, 80, T`
            if wav.ndim == 3 and wav.shape[1] == 1:
                wav = wav.squeeze(1)
            waveform_lengths = mel_lengths * vocoder.samples_per_frame
            audio = GeneratedAudio(
                waveforms=wav,
                lengths=waveform_lengths,
                sample_rate=vocoder.sample_rate,
            )
        else:
            audio = None

        return CosyVoiceGenerationOutput(mel=mel, mel_lengths=mel_lengths, audio=audio)

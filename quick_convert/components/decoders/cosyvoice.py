from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F
from torch import nn

from quick_convert.components.feature_extractors.mel import MelSpectrogramConfig
from quick_convert.data import GeneratedAudio
from quick_convert.utils.device import DeviceLike, configure_device
from quick_convert.utils.masking import make_padding_mask
from quick_convert.utils.sequence_validation import validate_feature_sequence

from .cosyvoice_reference import CosyVoiceReference, reference_flow_loss
from .stream_flow import content_and_speaker, independent_prefix_loss, stream_flow_loss, stream_flow_sample
from .streams import ConditioningStream, StreamConditioning


if TYPE_CHECKING:
    from quick_convert.components.encoders.upsample_conformer import UpsampleConformerEncoder

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
        stream_conditioning: StreamConditioning | None = None,
        stream_inference_steps: int = 10,
        content_encoder: UpsampleConformerEncoder | None = None,
    ):
        super().__init__()

        self.device = configure_device(device)
        self.flow = flow
        self.content_encoder = content_encoder
        self.feature_dim = feature_dim
        if content_encoder is not None:
            if feature_dim != content_encoder.input_size:
                raise ValueError("feature_dim must match the content encoder input_size.")
            if stream_conditioning is None or flow.vocab_size is not None or hasattr(flow, "encoder"):
                raise ValueError("A content_encoder requires a continuous DiT flow with named streams.")
            if content_encoder.upsample_factor != 1:
                raise ValueError("The duration-controlled content encoder requires upsample_factor=1.")
            # The first-party encoder replaces the donor's shallow content encoder.
            self.flow.pre_lookahead_layer = None
        self.stream_conditioning = stream_conditioning
        if type(stream_inference_steps) is not int or stream_inference_steps < 1:
            raise ValueError("stream_inference_steps must be a positive integer.")
        self.stream_inference_steps = stream_inference_steps
        if stream_conditioning is not None:
            if flow.vocab_size is not None or not (hasattr(flow, "pre_lookahead_layer") or hasattr(flow, "encoder")):
                raise ValueError("Named streams require a continuous-feature CosyVoice flow.")
            estimator = flow.decoder.estimator
            if getattr(estimator, "cond_dim", None) != stream_conditioning.output_dim:
                raise ValueError("Estimator cond_dim must equal stream_conditioning.output_dim.")
            if "mel" in stream_conditioning.dimensions and stream_conditioning.dimensions["mel"] != mel_dim:
                raise ValueError("The mel stream dimension must equal mel_dim.")
        if stream_conditioning is not None and not self.requires_speaker_embedding:
            # Remove unused trainable parameters as well as the runtime input.
            self.flow.spk_embed_affine_layer = None
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
        projection_dim = content_encoder.output_size() if content_encoder is not None else feature_dim
        if flow.vocab_size is None and projection_dim != flow.input_size:
            self.input_projection = nn.Linear(projection_dim, flow.input_size, device=self.device)

    def requested_mel_lengths(self, lengths, duration_ratio=1.0):
        """Convert source feature lengths and a duration multiplier to mel lengths."""
        ratio = torch.as_tensor(duration_ratio, device=lengths.device, dtype=torch.float32)
        if ratio.ndim > 1 or (ratio.ndim == 1 and ratio.shape != lengths.shape):
            raise ValueError("duration_ratio must be a scalar or one value per example.")
        if not torch.isfinite(ratio).all() or torch.any(ratio <= 0):
            raise ValueError("duration_ratio must be finite and positive.")
        if getattr(self, "content_encoder", None) is None and torch.any(ratio != 1):
            raise ValueError("duration_ratio requires a duration-controlled content encoder.")
        return (self.flow.output_lengths(lengths) * ratio).round().long().clamp_min(1)

    def _content_and_speaker(self, features, lengths, speaker, mel_lengths):
        if getattr(self, "content_encoder", None) is None:
            return content_and_speaker(self.flow, features, lengths, speaker)
        h, valid = self.content_encoder(features, lengths, target_lengths=mel_lengths)
        if self.input_projection is not None:
            h = self.input_projection(h)
        h = h.masked_fill(~valid.transpose(1, 2), 0)
        spks = None if speaker is None else self.flow.spk_embed_affine_layer(F.normalize(speaker, dim=1))
        return h, spks

    @property
    def requires_speaker_embedding(self) -> bool:
        """Whether this composition consumes a global speaker vector."""
        if getattr(self, "stream_conditioning", None) is None:
            return True
        estimator = self.flow.decoder.estimator
        embedding = getattr(estimator, "input_embed", estimator)
        return getattr(embedding, "spk_dim", 1) > 0

    def _validate_speaker_embedding(self, speaker_embedding):
        if self.requires_speaker_embedding and speaker_embedding is None:
            raise ValueError("This decoder requires a speaker_embedding.")
        if not self.requires_speaker_embedding and speaker_embedding is not None:
            raise ValueError("This decoder disables x-vector conditioning; omit speaker_embedding.")

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

    def prepare_reference(self, features, lengths, waveform, waveform_lengths, sample_rate) -> CosyVoiceReference:
        """Extract matching reference mels and trim extraction-boundary differences."""
        self._require_reference_flow()
        mels, mel_lengths = self._compute_mels(waveform, waveform_lengths, int(sample_rate))
        ratio = self.flow.token_mel_ratio
        if torch.any((lengths * ratio - mel_lengths).abs() > ratio):
            raise ValueError("Reference content and mel durations differ by more than one content frame.")
        lengths = torch.minimum(lengths, mel_lengths // ratio)
        reference = CosyVoiceReference(
            features[:, : int(lengths.max())],
            lengths,
            mels[:, :, : int(lengths.max()) * ratio].transpose(1, 2),
            lengths * ratio,
        )
        reference.validate(features.shape[0], features.shape[-1], self.mel_dim, ratio)
        return reference

    def _require_reference_flow(self):
        if self.flow.vocab_size is not None or not (
            hasattr(self.flow, "encoder") or hasattr(self.flow, "pre_lookahead_layer")
        ):
            raise ValueError("Reference prompts require a continuous-feature CosyVoice flow.")

    def _project_reference(self, reference, batch_size, feature_dim):
        self._require_reference_flow()
        reference.validate(batch_size, feature_dim, self.mel_dim, self.flow.token_mel_ratio)
        if self.input_projection is not None:
            reference = replace(reference, features=self.input_projection(reference.features))
        return reference

    def compute_loss(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        target_wav: torch.Tensor,
        wav_lens: torch.Tensor,
        sampling_rate: int,
        speaker_embedding: torch.Tensor | None = None,
        mask: torch.Tensor = None,
        reference: CosyVoiceReference | None = None,
        streams: dict[str, ConditioningStream] | None = None,
        # cond: Optional[torch.Tensor] = None,
    ):
        """
        Thin wrapper around donor compute_loss.
        """

        validate_feature_sequence(features, lengths, feature_dim=getattr(self, "feature_dim", None))
        self._validate_speaker_embedding(speaker_embedding)
        if streams is not None and self.stream_conditioning is None:
            raise ValueError("Named streams require stream_conditioning configuration.")
        if self.stream_conditioning is not None and reference is not None:
            raise ValueError("Use named stream prefixes instead of CosyVoiceReference in stream mode.")
        input_rate = getattr(self.flow, "input_frame_rate", None)
        ratio = getattr(self.flow, "token_mel_ratio", None)
        if input_rate is not None and ratio is not None:
            if abs(input_rate * ratio - self.mel_config.sample_rate / self.mel_config.hop_size) > 1e-6:
                raise ValueError("Flow output frame rate differs from the mel specification.")
        target_mel, target_mel_lengths = self._compute_mels(target_wav, wav_lens, int(sampling_rate))

        ratio = ratio if ratio is not None else 1
        if type(ratio) is not int or ratio < 1:
            raise ValueError("token_mel_ratio must be a positive integer.")
        if getattr(self, "content_encoder", None) is not None:
            if (
                lengths.shape != target_mel_lengths.shape
                or torch.any(lengths <= 0)
                or torch.any(lengths > features.shape[1])
            ):
                raise ValueError("Content lengths must match the batch and fit feature storage.")
            duration_ratio = target_mel_lengths.float() / self.flow.output_lengths(lengths)
            mel_lengths = self.requested_mel_lengths(lengths, duration_ratio)
        else:
            # Compare endpoints on the mel axis. Different extraction boundaries
            # may disagree by one content frame; do not hide larger rate errors.
            differences = (lengths * ratio - target_mel_lengths).abs()
            if torch.any(differences > ratio):
                raise ValueError(
                    f"Content and mel durations differ by more than one content frame: {differences.tolist()}."
                )
            lengths = torch.minimum(lengths, target_mel_lengths // ratio)
            if torch.any(lengths <= 0):
                raise ValueError("Reconstruction requires at least one aligned content frame per sample.")
            mel_lengths = lengths * ratio
        features = features[:, : int(lengths.max())]
        target_mel = target_mel[:, :, : int(mel_lengths.max())]

        if reference is not None:
            if not hasattr(self.flow, "encoder"):
                raise ValueError(
                    "Separate-reference training requires an encoder flow. "
                    "For native DiT prefix training, omit reference; the flow samples its own mel prefix."
                )
            reference = self._project_reference(reference, features.shape[0], features.shape[-1])

        if self.input_projection is not None and getattr(self, "content_encoder", None) is None:
            mask = make_padding_mask(lengths, max_length=features.shape[1])
            features = self.input_projection(features) * mask.unsqueeze(-1)

        if self.stream_conditioning is not None:
            targets = target_mel.transpose(1, 2)
            supplied = dict(streams or {})
            if "mel" in supplied:
                if getattr(self, "content_encoder", None) is None or not supplied["mel"].is_prefix:
                    raise ValueError("An external training mel stream requires a duration-controlled reference prefix.")
            elif "mel" in self.stream_conditioning.dimensions:
                supplied["mel"] = ConditioningStream(targets, mel_lengths)
            if getattr(self, "content_encoder", None) is not None:
                h, spks = self._content_and_speaker(features, lengths, speaker_embedding, mel_lengths)
                return CosyVoiceDecoderOutput(
                    loss=independent_prefix_loss(
                        self.flow, self.stream_conditioning, h, spks, targets, mel_lengths, supplied
                    )
                )
            conditions, loss_mask = self.stream_conditioning.training_conditions(supplied, mel_lengths, targets)
            h, spks = self._content_and_speaker(features, lengths, speaker_embedding, mel_lengths)
            loss = stream_flow_loss(self.flow, h, spks, targets, mel_lengths, conditions, loss_mask)
            return CosyVoiceDecoderOutput(loss=loss)

        batch = {
            "speech_token": features,
            "speech_token_len": lengths,
            "speech_feat": target_mel.transpose(1, 2),
            "speech_feat_len": mel_lengths,
            "embedding": speaker_embedding,
        }

        if reference is not None:
            loss = reference_flow_loss(
                self.flow, features, lengths, target_mel.transpose(1, 2), reference, speaker_embedding
            )
            return CosyVoiceDecoderOutput(loss=loss)

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
        speaker_embedding: torch.Tensor | None = None,
        run_vocoder: bool = False,
        reference: CosyVoiceReference | None = None,
        inference_cfg_rate: float | None = None,
        streams: dict[str, ConditioningStream] | None = None,
        duration_ratio: float | torch.Tensor = 1.0,
    ) -> CosyVoiceGenerationOutput:
        """Generate with optional per-call CFG; None retains the saved guidance rate."""
        validate_feature_sequence(feature, length, feature_dim=getattr(self, "feature_dim", None))
        self._validate_speaker_embedding(speaker_embedding)
        if inference_cfg_rate is not None:
            if not math.isfinite(inference_cfg_rate) or inference_cfg_rate < 0:
                raise ValueError("inference_cfg_rate must be finite and nonnegative.")
        if self.stream_conditioning is not None and reference is not None:
            raise ValueError("Use named stream prefixes instead of CosyVoiceReference in stream mode.")
        generation_kwargs = {} if inference_cfg_rate is None else {"inference_cfg_rate": inference_cfg_rate}
        length = length.clone()
        feature = feature[:, : int(length.max())]

        if reference is not None:
            reference = self._project_reference(reference, feature.shape[0], feature.shape[-1])
            if not torch.all(reference.lengths == reference.lengths[0]):
                raise ValueError("Batched generation requires equal reference lengths.")

        if self.input_projection is not None and getattr(self, "content_encoder", None) is None:
            mask = make_padding_mask(length, max_length=feature.shape[1])
            feature = self.input_projection(feature) * mask.unsqueeze(-1)

        mel_lengths = self.requested_mel_lengths(length, duration_ratio)
        if self.stream_conditioning is not None:
            h, spks = self._content_and_speaker(feature, length, speaker_embedding, mel_lengths)
            prefix_streams = streams or {}
            if torch.is_autocast_enabled(h.device.type):
                # Frozen extractors return float32 while the content encoder can
                # produce an autocast dtype during validation generation.
                prefix_streams = {
                    name: replace(stream, values=stream.values.to(dtype=h.dtype))
                    for name, stream in prefix_streams.items()
                }
            prefixes, prefix_lengths = self.stream_conditioning.prefix_conditions(prefix_streams, h)
            cfg_rate = self.flow.decoder.inference_cfg_rate if inference_cfg_rate is None else inference_cfg_rate
            mel = stream_flow_sample(
                self.flow,
                h,
                mel_lengths,
                spks,
                prefixes,
                prefix_lengths,
                self.stream_inference_steps,
                cfg_rate,
            )
        else:
            if streams is not None:
                raise ValueError("Named streams require stream_conditioning configuration.")
            B, _, D = feature.shape
            mel, _ = self.flow.inference(
                token=feature,
                token_len=length,
                prompt_token=reference.features[:, : int(reference.lengths[0])]
                if reference is not None
                else feature.new_zeros((B, 0, D)),
                prompt_token_len=reference.lengths if reference is not None else length.new_zeros((B,)),
                prompt_feat=reference.mels[:, : int(reference.mel_lengths[0])]
                if reference is not None
                else feature.new_zeros((B, 0, self.flow.output_size)),
                prompt_feat_len=reference.mel_lengths if reference is not None else length.new_zeros((B,)),
                embedding=speaker_embedding,
                finalize=True,
                streaming=False,
                **generation_kwargs,
                # max_feature_len=max_len,
                # n_timesteps=n_timesteps,
            )
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

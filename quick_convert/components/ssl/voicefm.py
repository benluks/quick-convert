from __future__ import annotations

import math
from pathlib import Path

import torch
import torchaudio

from ._waveforms import validate_waveforms
from .base import ContentEncoder, ContentFeatures


class VoiceFMContentEncoder(ContentEncoder):
    """Frame representations from a checkpoint's VoiceFM-Whisper backbone.

    Requires trained weights: a base Whisper checkpoint is not VoiceFM.
    Returns states before temporal pooling, task conditioning, and projection.
    ``layer=None`` preserves transformer layers for weighted-sum probing;
    integer indices address the Hugging Face hidden-state list, including the
    input embedding at index zero. Inputs longer than 30 seconds are rejected.
    """

    FEATURE_DIM = 1280
    SAMPLE_RATE = 16000
    MAX_SAMPLES = 480000

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_name: str = "openai/whisper-large-v2",
        layer: int | None = -1,
        device: str | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
    ) -> None:
        super().__init__(device)
        self.model_name = str(checkpoint_path)
        self.layer = layer
        # Fail before allocating a large encoder when the checkpoint is absent.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            raise TypeError("VoiceFM requires a tensor state dictionary or a training checkpoint containing one.")
        state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        if not isinstance(state, dict):
            raise TypeError("VoiceFM checkpoint does not contain a state dictionary.")
        for prefix in ("audio_encoder.encoder.", "encoder."):
            selected = {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}
            if selected:
                state = selected
                break

        from transformers import WhisperConfig, WhisperFeatureExtractor
        from transformers.models.whisper.modeling_whisper import WhisperEncoder

        config = WhisperConfig.from_pretrained(model_name, local_files_only=local_files_only, revision=revision)
        self.processor = WhisperFeatureExtractor.from_pretrained(
            model_name, local_files_only=local_files_only, revision=revision
        )
        if self.processor.sampling_rate != self.sample_rate or self.processor.n_samples != self.MAX_SAMPLES:
            raise ValueError("VoiceFM-Whisper requires the 16 kHz, 30-second Whisper frontend.")
        self.model = WhisperEncoder(config)
        # All encoder weights are required, including the frozen lower layers.
        # No partial matching or fallback to unmodified Whisper weights.
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.to(self.device).eval()

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    @property
    def feature_dim(self) -> int:
        return self.model.config.d_model

    @property
    def N_LAYERS(self) -> int:
        return self.model.config.encoder_layers

    @property
    def frame_hz(self) -> float:
        return self.sample_rate / (
            self.processor.hop_length * math.prod((self.model.conv1.stride[0], self.model.conv2.stride[0]))
        )

    def encode_file(self, path: str | Path) -> ContentFeatures:
        waveform, sr = torchaudio.load(str(path))
        waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        return self.encode_waveforms(waveform)

    def forward(self, batch) -> ContentFeatures:
        if batch.waveforms is None:
            raise RuntimeError("VoiceFM requires loaded audio; set load=True in the dataset.")
        if not (batch.sample_rates == self.sample_rate).all():
            raise ValueError(f"Set target_sr={self.sample_rate} in the dataset.")
        return self.encode_waveforms(batch.waveforms, batch.lengths)

    @torch.inference_mode()
    def encode_waveforms(self, waveforms, lengths=None, sample_rate=None) -> ContentFeatures:
        lengths = validate_waveforms(waveforms, lengths, sample_rate, self.sample_rate)
        if torch.any(lengths > self.MAX_SAMPLES):
            raise ValueError("VoiceFM-Whisper accepts at most 30 seconds per item; split longer recordings first.")
        items = [waveforms[i, : int(length)].detach().cpu().numpy().copy() for i, length in enumerate(lengths)]
        inputs = self.processor(
            items,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
            padding="max_length",
            max_length=self.MAX_SAMPLES,
            truncation=False,
        )
        outputs = self.model(inputs.input_features.to(self.device), output_hidden_states=True, return_dict=True)
        if self.layer is None:
            values = torch.stack(outputs.hidden_states[1:], dim=2)
        else:
            try:
                values = outputs.hidden_states[self.layer]
            except IndexError as error:
                raise ValueError(
                    f"Invalid VoiceFM layer {self.layer}; returned {len(outputs.hidden_states)} hidden states."
                ) from error
        # The feature extractor's mask is already at mel-frame resolution.
        output_lengths = inputs.attention_mask.sum(dim=-1).to(values.device, dtype=torch.long)
        for convolution in (self.model.conv1, self.model.conv2):
            output_lengths = (
                torch.div(
                    output_lengths
                    + 2 * convolution.padding[0]
                    - convolution.dilation[0] * (convolution.kernel_size[0] - 1)
                    - 1,
                    convolution.stride[0],
                    rounding_mode="floor",
                )
                + 1
            )
        values = values[:, : int(output_lengths.max())]
        return ContentFeatures(
            values=values,
            lengths=output_lengths,
            feature_dim=values.shape[-1],
            representation_type="continuous",
            temporal_granularity="frame",
            backend="transformers",
            model_name=self.model_name,
            layer=self.layer,
            frame_hz=self.frame_hz,
        )

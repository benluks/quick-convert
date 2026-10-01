from __future__ import annotations

from pathlib import Path

import torch
import torchaudio

from ._waveforms import validate_waveforms
from .base import ContentEncoder, ContentFeatures


class SPEARContentEncoder(ContentEncoder):
    """Official SPEAR v2 waveform encoder, including aligned intermediate layers.

    ``layer=None`` retains all layers as ``(batch, frames, layers, features)``.
    Integer indices select the upstream hidden-state list (zero based).
    The default XLarge speech/audio checkpoint has 1280-dimensional features.
    SPEAR's Hugging Face release requires its checkpoint's remote Python code.
    """

    FEATURE_DIM = 1280
    SAMPLE_RATE = 16000
    FRAME_HZ = 50.0

    def __init__(
        self,
        model_name: str = "marcoyang/spear-xlarge-speech-audio-v2",
        layer: int | None = -1,
        device: str | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__(device)
        self.model_name = model_name
        self.layer = layer
        from transformers import AutoModel

        self.model = (
            AutoModel.from_pretrained(
                model_name, local_files_only=local_files_only, revision=revision, trust_remote_code=trust_remote_code
            )
            .to(self.device)
            .eval()
        )
        self._num_layers = sum(int(value) for value in self.model.config.num_encoder_layers.split(","))
        dimensions = self.model.config.encoder_dim
        self._feature_dim = max(int(value) for value in dimensions.split(","))
        if int(self.model.config.output_downsampling_factor) != 1:
            raise ValueError("This adapter requires SPEAR v2 checkpoints with output_downsampling_factor=1.")

    @property
    def N_LAYERS(self) -> int:
        """Checkpoint layer count, exposed for downstream weighted-sum fusion."""
        return self._num_layers

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    @property
    def frame_hz(self) -> float:
        return self.FRAME_HZ

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def encode_file(self, path: str | Path) -> ContentFeatures:
        waveform, sr = torchaudio.load(str(path))
        waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        return self.encode_waveforms(waveform)

    def forward(self, batch) -> ContentFeatures:
        if batch.waveforms is None:
            raise RuntimeError("SPEAR requires loaded audio; set load=True in the dataset.")
        if not (batch.sample_rates == self.sample_rate).all():
            raise ValueError(f"Set target_sr={self.sample_rate} in the dataset.")
        return self.encode_waveforms(batch.waveforms, batch.lengths)

    @torch.inference_mode()
    def encode_waveforms(self, waveforms, lengths=None, sample_rate=None) -> ContentFeatures:
        lengths = validate_waveforms(waveforms, lengths, sample_rate, self.sample_rate)
        outputs = self.model(waveforms.to(self.device), lengths.to(self.device))
        states = outputs["hidden_states"]
        if not states:
            raise RuntimeError("SPEAR did not return intermediate hidden states.")
        if self.layer is None:
            if len({tuple(state.shape) for state in states}) != 1:
                raise ValueError("All-layer SPEAR extraction requires aligned states with equal feature dimensions.")
            values = torch.stack(states, dim=2)
        else:
            try:
                values = states[self.layer]
            except IndexError as error:
                raise ValueError(
                    f"Invalid SPEAR layer {self.layer}; checkpoint returned {len(states)} states."
                ) from error
        return ContentFeatures(
            values=values,
            lengths=outputs["encoder_out_lens"].to(values.device, dtype=torch.long),
            feature_dim=values.shape[-1],
            representation_type="continuous",
            temporal_granularity="frame",
            backend="transformers",
            model_name=self.model_name,
            layer=self.layer,
            frame_hz=self.frame_hz,
        )

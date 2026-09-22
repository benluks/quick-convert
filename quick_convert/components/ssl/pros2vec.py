from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.nn.utils.rnn import pad_sequence

from quick_convert.data.types import AudioBatch

from .base import ContentEncoder, ContentFeatures


# Source: https://github.com/MiniXC/masked_prosody_model/tree/main


class ProsodyEncoder(ContentEncoder):
    """Frame-level prosody representations from Masked Prosody Model."""

    HOP_LENGTH = 256
    POOL_FACTOR = 2
    WINDOW_SECONDS = 6

    def __init__(
        self,
        model_name: str = "cdminix/masked_prosody_model",
        sample_rate: int = 22_050,
        layer: int = 7,
        device: str | None = None,
    ) -> None:
        super().__init__(device=device)
        self.model_name = model_name
        self._sample_rate = sample_rate
        self.layer = layer

        from masked_prosody_model import MaskedProsodyModel

        self.model = MaskedProsodyModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.FEATURE_DIM = int(self.model.args.filter_size)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_hz(self) -> float:
        return self.sample_rate / (self.HOP_LENGTH * self.POOL_FACTOR)

    def encode_file(self, path: str | Path) -> ContentFeatures:
        """Load an audio file, downmix it, and encode its valid samples."""
        waveform, sample_rate = torchaudio.load(Path(path))
        if waveform.ndim != 2:
            raise ValueError(f"Expected waveform with shape (channels, time), got {tuple(waveform.shape)}.")
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return self.encode_waveforms(waveform, sample_rate=sample_rate)

    @torch.inference_mode()
    def encode_waveforms(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor | None = None,
        sample_rate: int | None = None,
    ) -> ContentFeatures:
        """Encode padded waveforms while respecting each sample's valid length."""
        if waveforms.ndim != 2:
            raise ValueError(f"Expected waveforms with shape (batch, time), got {tuple(waveforms.shape)}.")

        batch_size, padded_length = waveforms.shape
        if lengths is None:
            lengths = torch.full((batch_size,), padded_length, dtype=torch.long, device=waveforms.device)
        if lengths.ndim != 1 or lengths.shape[0] != batch_size:
            raise ValueError("lengths must contain one value per waveform.")
        if torch.any(lengths <= 0) or torch.any(lengths > padded_length):
            raise ValueError(f"lengths must be between 1 and the padded length ({padded_length}).")

        input_sample_rate = sample_rate or self.sample_rate
        outputs = []
        processed_lengths = []
        for waveform, length in zip(waveforms, lengths, strict=True):
            waveform = waveform[: int(length)].detach().cpu()
            if input_sample_rate != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, input_sample_rate, self.sample_rate)
            outputs.append(self._encode_waveform(waveform))
            processed_lengths.append(waveform.shape[0])

        output_lengths = self.output_lengths(torch.tensor(processed_lengths, dtype=torch.long, device=self.device))
        observed_lengths = torch.tensor([output.shape[0] for output in outputs], dtype=torch.long, device=self.device)
        if not torch.equal(output_lengths, observed_lengths):
            raise RuntimeError(
                "Masked Prosody Model output disagrees with its deterministic length calculation: "
                f"expected {output_lengths.tolist()}, observed {observed_lengths.tolist()}."
            )
        values = pad_sequence(outputs, batch_first=True)

        return ContentFeatures(
            values=values,
            lengths=output_lengths,
            feature_dim=self.feature_dim,
            representation_type="continuous",
            temporal_granularity="frame",
            backend="masked-prosody-model",
            model_name=self.model_name,
            layer=self.layer,
            frame_hz=self.frame_hz,
        )

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Calculate pooled MPM frame counts, including six-second chunk boundaries."""
        window_size = self.sample_rate * self.WINDOW_SECONDS
        full_windows = torch.div(input_lengths, window_size, rounding_mode="floor")
        remainder = torch.remainder(input_lengths, window_size)

        frames_per_window = window_size // self.HOP_LENGTH + 1
        raw_frames = full_windows * frames_per_window
        raw_frames += torch.where(
            remainder > 0,
            torch.div(remainder, self.HOP_LENGTH, rounding_mode="floor") + 1,
            0,
        )
        return torch.div(raw_frames + self.POOL_FACTOR - 1, self.POOL_FACTOR, rounding_mode="floor")

    def _encode_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        audio = waveform.numpy()
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak

        representations = []
        window_size = self.sample_rate * self.WINDOW_SECONDS
        for start in range(0, len(audio), window_size):
            representations.append(self._encode_window(audio[start : start + window_size]))

        representation = torch.cat(representations, dim=0)
        return (
            F.avg_pool1d(
                representation.transpose(0, 1).unsqueeze(0),
                kernel_size=3,
                stride=self.POOL_FACTOR,
                padding=1,
            )
            .squeeze(0)
            .transpose(0, 1)
        )

    def _encode_window(self, window: np.ndarray) -> torch.Tensor:
        durations = np.array([1000])
        pitch = self.model.pitch_measure(window, durations)["measure"]
        energy = self.model.energy_measure(window, durations)["measure"]
        vad = self.model.vad_measure(window, durations)["measure"]

        frame_count = min(len(pitch), len(energy), len(vad))
        pitch = self._normalize_measure(pitch[:frame_count], self.model.args.pitch_min, self.model.args.pitch_max)
        energy = self._normalize_measure(energy[:frame_count], self.model.args.energy_min, self.model.args.energy_max)
        vad = self._normalize_measure(vad[:frame_count], self.model.args.vad_min, self.model.args.vad_max)

        bins = self.model.bins.to(self.device)
        features = torch.stack(
            [
                torch.bucketize(torch.as_tensor(pitch, device=self.device), bins),
                torch.bucketize(torch.as_tensor(energy, device=self.device), bins),
                torch.bucketize(
                    torch.as_tensor(vad, device=self.device),
                    torch.linspace(0, 1, 2, device=self.device),
                ),
            ],
        ).long()

        result = self.model(features.unsqueeze(0), return_layer=self.layer)
        representation = result["representations"]
        if representation is None:
            raise ValueError(f"Masked Prosody Model has no representation for layer {self.layer}.")
        return representation.squeeze(0)

    @staticmethod
    def _normalize_measure(values: np.ndarray, minimum: float, maximum: float) -> np.ndarray:
        values = np.nan_to_num(values, nan=minimum)
        return np.clip(values, minimum, maximum) / (maximum - minimum)

    def forward(self, batch: AudioBatch) -> ContentFeatures:
        if batch.waveforms is None or batch.sample_rates is None:
            raise RuntimeError(f"{type(self).__name__} requires a batch with loaded audio.")
        if not (batch.sample_rates == self.sample_rate).all():
            raise RuntimeError(
                f"Expected {self.sample_rate} Hz audio, got {batch.sample_rates}. "
                f"Set target_sr={self.sample_rate} in the dataset configuration."
            )
        return self.encode_waveforms(batch.waveforms, batch.lengths, self.sample_rate)

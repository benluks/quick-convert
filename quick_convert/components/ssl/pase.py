from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

from ._waveforms import validate_waveforms
from .base import ContentEncoder, ContentFeatures


class PASEContentEncoder(ContentEncoder):
    """Official PASE/PASE+ frontend using a local config and pretrained checkpoint.

    Encodes each unpadded waveform separately: the recurrent frontend and
    temporal normalization must not see another item's padding. All-layer mode
    uses pretrained dense skip projections aligned by the upstream frontend.
    """

    FEATURE_DIM = 256

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: str | None = None,
        layer: int | None = -1,
    ) -> None:
        super().__init__(device)
        if layer is not None and (not isinstance(layer, int) or isinstance(layer, bool) or layer < -1):
            raise ValueError(
                "Use layer=-1 for final features, layer=None for all stages, or a nonnegative stage index."
            )
        self.layer = layer
        self.model_name = str(checkpoint_path)
        with Path(config_path).open() as file:
            config = json.load(file)
        self._sample_rate = int(config.get("sr", 16000))
        self._frame_hz = self._sample_rate / math.prod(config.get("strides", [1, 10, 2, 1, 2, 1, 2, 2]))
        from pase.models.frontend import wf_builder

        self.model = wf_builder(config)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("state_dict", checkpoint)
        # Strict loading prevents accidentally probing a partially random model.
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.to(self.device).eval()
        self._feature_dim = self.model.emb_dim
        if layer != -1:
            if (
                not hasattr(self.model, "denseskips")
                or not len(self.model.denseskips)
                or getattr(self.model, "densemerge", None) != "sum"
                or not callable(getattr(self.model, "fuse_skip", None))
            ):
                raise ValueError(
                    "PASE all-layer extraction requires pretrained dense skip projections with sum merging."
                )
            if layer is not None and layer >= self.N_LAYERS:
                raise ValueError(f"Invalid PASE layer {layer}; frontend exposes {self.N_LAYERS} stages.")

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_hz(self) -> float:
        return self._frame_hz

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def N_LAYERS(self) -> int | None:
        """Projected skip stages followed by the final aggregate frontend output."""
        skips = getattr(self.model, "denseskips", None)
        return len(skips) + 1 if skips is not None and len(skips) else None

    def _extract_stages(self, source: torch.Tensor) -> torch.Tensor:
        skips = []

        def capture(_module, _inputs, output):
            skips.append(output.clone())

        handles = []
        try:
            for projection in self.model.denseskips:
                handles.append(projection.register_forward_hook(capture))
            final = self.model(source)
        finally:
            for handle in handles:
                handle.remove()
        if len(skips) != len(self.model.denseskips):
            raise RuntimeError("PASE did not execute every dense skip projection.")
        # Reuse the exact crop/mean downsampling used when the checkpoint sums skips.
        states = [self.model.fuse_skip(torch.zeros_like(final), skip) for skip in skips]
        states.append(final)
        if any(state.shape != final.shape for state in states):
            raise RuntimeError("PASE projected stages are not aligned with final audio frames.")
        if self.layer is None:
            return torch.stack([state.squeeze(0).transpose(0, 1) for state in states], dim=1)
        return states[self.layer].squeeze(0).transpose(0, 1)

    def encode_file(self, path: str | Path) -> ContentFeatures:
        waveform, sr = torchaudio.load(str(path))
        waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        return self.encode_waveforms(waveform)

    def forward(self, batch) -> ContentFeatures:
        if batch.waveforms is None:
            raise RuntimeError("PASE requires loaded audio; set load=True in the dataset.")
        if not (batch.sample_rates == self.sample_rate).all():
            raise ValueError(f"Set target_sr={self.sample_rate} in the dataset.")
        return self.encode_waveforms(batch.waveforms, batch.lengths)

    @torch.inference_mode()
    def encode_waveforms(self, waveforms, lengths=None, sample_rate=None) -> ContentFeatures:
        lengths = validate_waveforms(waveforms, lengths, sample_rate, self.sample_rate)
        features = []
        for waveform, length in zip(waveforms, lengths, strict=True):
            source = waveform[: int(length)].to(self.device)[None, None, :]
            if self.layer != -1:
                features.append(self._extract_stages(source))
                continue
            values = self.model(source)
            if not isinstance(values, torch.Tensor) or values.ndim != 3 or values.shape[0] != 1:
                raise RuntimeError("Expected a PASE frontend tensor with shape (1, features, frames).")
            features.append(values.squeeze(0).transpose(0, 1))
        values = pad_sequence(features, batch_first=True)
        return ContentFeatures(
            values=values,
            lengths=torch.tensor([len(item) for item in features], device=values.device),
            feature_dim=values.shape[-1],
            representation_type="continuous",
            temporal_granularity="frame",
            backend="pase",
            model_name=self.model_name,
            layer=self.layer,
            frame_hz=self.frame_hz,
        )

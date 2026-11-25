from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from quick_convert.components.encoders.content._waveforms import validate_waveforms
from quick_convert.components.encoders.content.base import ContentEncoder, ContentFeatures
from quick_convert.data.types import AudioBatch
from quick_convert.utils.audio import load_audio


class OriginalWavLMContentEncoder(ContentEncoder):
    """Extract features using the original WavLM-Large checkpoint and inference code.

    Encode each valid waveform separately to preserve upstream kNN-VC's
    unpadded, unnormalized feature extraction.
    """

    FEATURE_DIM = 1024
    SAMPLE_RATE = 16000
    FRAME_HZ = 50.0
    NUM_LAYERS = 24

    def __init__(self, layer: int = 6, device: str | None = None) -> None:
        super().__init__(device=device)
        if not 1 <= layer <= self.NUM_LAYERS:
            raise ValueError(f"Expected a WavLM layer from 1 to {self.NUM_LAYERS}, got {layer}.")
        self.layer = layer
        self.model = torch.hub.load("bshall/knn-vc", "wavlm_large", trust_repo=True, device=str(self.device)).eval()

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    @property
    def frame_hz(self) -> float:
        return self.FRAME_HZ

    def forward(self, batch: AudioBatch) -> ContentFeatures:
        if batch.waveforms is None or batch.lengths is None or batch.sample_rates is None:
            raise ValueError("Original WavLM requires loaded audio.")
        if not (batch.sample_rates == self.sample_rate).all():
            raise ValueError(f"Original WavLM requires {self.sample_rate} Hz audio.")
        return self.encode_waveforms(batch.waveforms, batch.lengths)

    def encode_file(self, path: str | Path) -> ContentFeatures:
        waveform, sample_rate = load_audio(path, target_sr=self.sample_rate, mono=True, device="cpu")
        return self.encode_waveforms(waveform, sample_rate=sample_rate)

    def encode_waveforms(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor | None = None,
        sample_rate: int | None = None,
    ) -> ContentFeatures:
        lengths = validate_waveforms(waveforms, lengths, sample_rate, self.sample_rate)
        device = next(self.model.parameters()).device
        features = [
            self.model.extract_features(waveform[None, :length].to(device), output_layer=self.layer)[0][0]
            for waveform, length in zip(waveforms, lengths.tolist(), strict=True)
        ]
        values = pad_sequence(features, batch_first=True)
        return ContentFeatures(
            values=values,
            lengths=torch.tensor([feature.shape[0] for feature in features], device=values.device),
            feature_dim=self.feature_dim,
            representation_type="continuous",
            temporal_granularity="frame",
            backend="original_wavlm",
            model_name="WavLM-Large",
            layer=self.layer,
            frame_hz=self.frame_hz,
        )

from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from quick_convert.components.encoders.content._waveforms import validate_waveforms
from quick_convert.components.encoders.content.base import ContentEncoder, ContentFeatures
from quick_convert.data.types import AudioBatch
from quick_convert.utils.audio import load_audio


# The Google Drive download linked in Microsoft's WavLM README.
_CHECKPOINT_URL = (
    "https://drive.usercontent.google.com/download?id=12-cB34qCTvByWT-QtOcZaqwwO21FLSqU&export=download&confirm=t"
)
_CACHE_NAME = "microsoft-wavlm-large-6fb4b3c3e6aa567f0a997b30855859cb81528ee8078802af439f7b2da0bf100f.pt"


class MicrosoftWavLMContentEncoder(ContentEncoder):
    """Extract features with Microsoft's WavLM-Large implementation and weights."""

    FEATURE_DIM = 1024
    SAMPLE_RATE = 16000
    FRAME_HZ = 50.0
    NUM_LAYERS = 24

    def __init__(self, layer: int = 6, device: str | None = None, *, do_normalize: bool = True) -> None:
        super().__init__(device=device)
        if not 1 <= layer <= self.NUM_LAYERS:
            raise ValueError(f"Expected a WavLM layer from 1 to {self.NUM_LAYERS}, got {layer}.")
        self.layer = layer
        self.do_normalize = do_normalize
        self.model = self._load_model().to(self.device).eval()

    @staticmethod
    def _load_model() -> torch.nn.Module:
        from quick_convert.external.wavlm.model import WavLM, WavLMConfig

        checkpoint = torch.hub.load_state_dict_from_url(
            _CHECKPOINT_URL,
            file_name=_CACHE_NAME,
            check_hash=True,
            map_location="cpu",
            weights_only=True,
        )
        model = WavLM(WavLMConfig(checkpoint["cfg"]))
        model.load_state_dict(checkpoint["model"])
        return model

    @property
    def sample_rate(self) -> int:
        return self.SAMPLE_RATE

    @property
    def frame_hz(self) -> float:
        return self.FRAME_HZ

    def forward(self, batch: AudioBatch) -> ContentFeatures:
        if batch.waveforms is None or batch.lengths is None or batch.sample_rates is None:
            raise ValueError("Microsoft WavLM requires loaded audio.")
        if not (batch.sample_rates == self.sample_rate).all():
            raise ValueError(f"Microsoft WavLM requires {self.sample_rate} Hz audio.")
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
        features = []
        for waveform, length in zip(waveforms, lengths.tolist(), strict=True):
            waveform = waveform[None, :length].to(device)
            if self.do_normalize:
                waveform = F.layer_norm(waveform, (length,))
            features.append(self.model.extract_features(waveform, output_layer=self.layer)[0][0])
        values = pad_sequence(features, batch_first=True)
        return ContentFeatures(
            values=values,
            lengths=torch.tensor([feature.shape[0] for feature in features], device=values.device),
            feature_dim=self.feature_dim,
            representation_type="continuous",
            temporal_granularity="frame",
            backend="microsoft",
            model_name="microsoft/wavlm-large",
            layer=self.layer,
            frame_hz=self.frame_hz,
        )

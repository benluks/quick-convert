import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from torch import nn

from quick_convert.components.ssl.pros2vec import ProsodyEncoder


CONFIG_DIR = Path(__file__).parents[1] / "configs"


class FakeMeasure:
    def __init__(self, value: float):
        self.value = value
        self.input_lengths = []

    def __call__(self, audio, durations):
        self.input_lengths.append(len(audio))
        frame_count = len(audio) // ProsodyEncoder.HOP_LENGTH + 1
        return {"measure": np.full(frame_count, self.value, dtype=np.float32)}


class FakeMaskedProsodyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.args = SimpleNamespace(
            filter_size=4,
            pitch_min=50,
            pitch_max=300,
            energy_min=0,
            energy_max=0.2,
            vad_min=0,
            vad_max=1,
        )
        self.bins = torch.linspace(0, 1, 8)
        self.pitch_measure = FakeMeasure(100)
        self.energy_measure = FakeMeasure(0.1)
        self.vad_measure = FakeMeasure(1)

    @classmethod
    def from_pretrained(cls, model_name):
        return cls()

    def forward(self, features, return_layer=None):
        assert features.ndim == 3
        assert features.shape[1] == 3
        frame_count = features.shape[2]
        representation = torch.arange(frame_count * 4, device=features.device).reshape(1, frame_count, 4).float()
        return {"representations": representation}


def test_prosody_encoder_respects_lengths_and_returns_padded_features(monkeypatch):
    fake_module = SimpleNamespace(MaskedProsodyModel=FakeMaskedProsodyModel)
    monkeypatch.setitem(sys.modules, "masked_prosody_model", fake_module)

    encoder = ProsodyEncoder(device="cpu")
    features = encoder.encode_waveforms(
        torch.arange(2048, dtype=torch.float32).reshape(2, 1024),
        lengths=torch.tensor([1024, 600]),
        sample_rate=22_050,
    )

    assert features.values.shape == (2, 3, 4)
    assert features.lengths.tolist() == [3, 2]
    assert features.feature_dim == 4
    assert features.backend == "masked-prosody-model"
    assert features.layer == 7
    assert encoder.model.pitch_measure.input_lengths == [1024, 600]


def test_content_encoder_default_device_is_usable(monkeypatch):
    fake_module = SimpleNamespace(MaskedProsodyModel=FakeMaskedProsodyModel)
    monkeypatch.setitem(sys.modules, "masked_prosody_model", fake_module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    encoder = ProsodyEncoder()

    assert encoder.device == torch.device("cpu")


def test_prosody_encoder_reports_chunk_aware_output_lengths():
    encoder = ProsodyEncoder.__new__(ProsodyEncoder)
    encoder.sample_rate = 22_050

    window = encoder.sample_rate * encoder.WINDOW_SECONDS
    lengths = encoder.output_lengths(torch.tensor([1, 256, 257, window, window + 1]))

    assert lengths.tolist() == [1, 1, 1, 259, 259]


def test_prosody_precompute_config_composes():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        config = compose(config_name="run/precompute_prosody_mpm_librispeech")

    assert config.encoder._target_ == "quick_convert.components.ssl.ProsodyEncoder"
    assert config.dataset.target_sr == 22_050
    assert config.pipeline.extractor.encoder is config.encoder

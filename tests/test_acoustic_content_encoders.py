import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from quick_convert.components.ssl import (
    EmotionEncoder,
    PASEContentEncoder,
    SPEARContentEncoder,
    resolve_content_encoder,
)


class FakeSPEAR(nn.Module):
    config = SimpleNamespace(encoder_dim="4,4", num_encoder_layers="1,1", output_downsampling_factor=1)

    def forward(self, audio, lengths):
        values = torch.zeros(audio.shape[0], 5, 4, device=audio.device)
        return {"hidden_states": [values, values + 1], "encoder_out_lens": lengths // 320}


@pytest.fixture
def spear_factory(monkeypatch):
    calls = []

    def load(name, **kwargs):
        calls.append((name, kwargs))
        return FakeSPEAR()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModel=SimpleNamespace(from_pretrained=load)))
    return calls


def test_spear_retains_all_layers_and_backend_lengths(spear_factory):
    encoder = SPEARContentEncoder(layer=None, device="cpu", local_files_only=True, revision="pinned")
    audio = torch.randn(2, 1600)
    lengths = torch.tensor([1600, 960])
    original = audio.clone()
    output = encoder.encode_waveforms(audio, lengths)
    assert output.values.shape == (2, 5, 2, 4)
    assert output.lengths.tolist() == [5, 3]
    assert output.feature_dim == encoder.feature_dim == 4
    assert output.frame_hz == 50
    assert encoder.N_LAYERS == output.values.shape[2]
    assert torch.equal(audio, original)
    assert spear_factory[0][1] == {"local_files_only": True, "revision": "pinned", "trust_remote_code": True}


def test_spear_last_layer_and_omitted_lengths(spear_factory):
    encoder = SPEARContentEncoder(device="cpu")
    output = encoder.encode_waveforms(torch.randn(1, 1600))
    assert output.lengths.tolist() == [5]
    assert torch.all(output.values == 1)
    encoder.layer = 7
    with pytest.raises(ValueError, match="Invalid SPEAR layer"):
        encoder.encode_waveforms(torch.randn(1, 1600))


@pytest.mark.parametrize(
    "lengths", [torch.tensor([0]), torch.tensor([1601]), torch.tensor([100.0]), torch.tensor([100, 100])]
)
def test_spear_rejects_invalid_lengths(spear_factory, lengths):
    encoder = SPEARContentEncoder(device="cpu")
    with pytest.raises((ValueError, TypeError)):
        encoder.encode_waveforms(torch.randn(1, 1600), lengths)


def test_spear_rejects_wrong_sample_rate(spear_factory):
    with pytest.raises(ValueError, match="16000"):
        SPEARContentEncoder(device="cpu").encode_waveforms(torch.randn(1, 1600), sample_rate=8000)


class FakePASE(nn.Module):
    emb_dim = 4

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.seen_lengths = []

    def forward(self, audio):
        self.seen_lengths.append(audio.shape[-1])
        return torch.ones(1, 4, audio.shape[-1] // 160) * self.weight


def test_pase_encodes_unpadded_items_and_loads_strictly(monkeypatch, tmp_path):
    model = FakePASE()
    monkeypatch.setitem(sys.modules, "pase", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "pase.models", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "pase.models.frontend", SimpleNamespace(wf_builder=lambda config: model))
    config = tmp_path / "frontend.json"
    config.write_text('{"strides": [10, 16]}')
    checkpoint = tmp_path / "weights.pt"
    torch.save({"weight": torch.tensor([2.0])}, checkpoint)
    encoder = PASEContentEncoder(config, checkpoint, device="cpu")
    output = encoder.encode_waveforms(torch.randn(2, 1600), torch.tensor([1600, 800]))
    assert model.seen_lengths == [1600, 800]
    assert output.values.shape == (2, 10, 4)
    assert output.lengths.tolist() == [10, 5]
    assert torch.all(output.values[0] == 2)
    assert torch.all(output.values[1, 5:] == 0)
    assert encoder.frame_hz == 100
    torch.save({}, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        PASEContentEncoder(config, checkpoint, device="cpu")


def test_emotion2vec_defaults_lengths_preserves_input_and_returns_features(monkeypatch):
    class FakeFunASR:
        model = nn.Identity()

        def generate(self, input, **kwargs):
            assert kwargs == {"granularity": "frame", "extract_embedding": True}
            return [{"feats": np.ones((len(waveform) // 320, 1024), dtype=np.float32)} for waveform in input]

    monkeypatch.setitem(sys.modules, "funasr", SimpleNamespace(AutoModel=lambda **kwargs: FakeFunASR()))
    encoder = EmotionEncoder(device="cpu")
    audio = torch.randn(2, 1600, requires_grad=True)
    output = encoder.encode_waveforms(audio)
    assert output.values.shape == (2, 5, 1024)
    assert output.lengths.tolist() == [5, 5]
    output = encoder.encode_waveforms(audio, torch.tensor([1600, 960]))
    assert output.lengths.tolist() == [5, 3]
    with pytest.raises(ValueError, match="16000"):
        encoder.encode_waveforms(audio, sample_rate=8000)


@pytest.mark.parametrize("kwargs", [
        {"layer": -2},
        {"layer": True},
        {"layer": None, "granularity": "utterance"},
        {"local_files_only": True},
        {"sample_rate": 8000},
    ])
def test_emotion2vec_rejects_unsupported_options(kwargs):
    with pytest.raises(ValueError):
        EmotionEncoder(device="cpu", **kwargs)


def test_new_encoder_aliases():
    assert resolve_content_encoder("spear") is SPEARContentEncoder
    assert resolve_content_encoder("pase") is PASEContentEncoder
    assert resolve_content_encoder("paseplus") is PASEContentEncoder

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from quick_convert.components.ssl import VoiceFMContentEncoder, resolve_content_encoder


class FakeEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv1d(80, 4, 3, padding=1)
        self.conv2 = nn.Conv1d(4, 4, 3, padding=1, stride=2)

    def forward(self, features, **kwargs):
        values = torch.zeros(features.shape[0], 1500, 4, device=features.device)
        return SimpleNamespace(hidden_states=(values, values + 1, values + 2))


class FakeProcessor:
    sampling_rate = 16000
    n_samples = 480000
    hop_length = 160

    def __call__(self, items, **kwargs):
        assert kwargs["truncation"] is False
        assert kwargs["return_attention_mask"] is True
        lengths = torch.tensor([(len(item) + 159) // 160 for item in items])
        return SimpleNamespace(
            input_features=torch.zeros(len(items), 80, 3000),
            attention_mask=torch.arange(3000)[None] < lengths[:, None],
        )


@pytest.fixture
def checkpoint(monkeypatch, tmp_path):
    config = SimpleNamespace(d_model=4, encoder_layers=2)
    module = SimpleNamespace(
        WhisperConfig=SimpleNamespace(from_pretrained=lambda *args, **kwargs: config),
        WhisperFeatureExtractor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeProcessor()),
    )
    monkeypatch.setitem(sys.modules, "transformers", module)
    monkeypatch.setitem(
        sys.modules, "transformers.models.whisper.modeling_whisper", SimpleNamespace(WhisperEncoder=FakeEncoder)
    )
    path = tmp_path / "voicefm.pt"
    state = {"audio_encoder.encoder." + key: value for key, value in FakeEncoder(config).state_dict().items()}
    state["clinical_encoder.unused"] = torch.zeros(1)
    torch.save({"model_state_dict": state}, path)
    return path


def test_voicefm_layers_exact_lengths_and_input_preservation(checkpoint):
    encoder = VoiceFMContentEncoder(checkpoint, layer=None, device="cpu")
    audio = torch.randn(2, 16000)
    original = audio.clone()
    result = encoder.encode_waveforms(audio, torch.tensor([16000, 12001]))
    assert result.values.shape == (2, 50, 2, 4)
    assert result.lengths.tolist() == [50, 38]
    assert result.frame_hz == 50
    assert encoder.N_LAYERS == 2
    assert encoder.feature_dim == 4
    assert torch.equal(audio, original)
    encoder.layer = -1
    result = encoder.encode_waveforms(audio)
    assert result.values.shape == (2, 50, 4)
    assert torch.all(result.values == 2)


def test_voicefm_missing_checkpoint_fails_before_loading_models(tmp_path):
    with pytest.raises(FileNotFoundError):
        VoiceFMContentEncoder(tmp_path / "absent.pt", device="cpu")


def test_voicefm_rejects_partial_encoder_weights(checkpoint):
    torch.save({"model_state_dict": {"audio_encoder.encoder.conv1.weight": torch.zeros(4, 80, 3)}}, checkpoint)
    with pytest.raises(RuntimeError, match="Missing key"):
        VoiceFMContentEncoder(checkpoint, device="cpu")


@pytest.mark.parametrize("prefix", ["", "encoder."])
def test_voicefm_accepts_encoder_only_and_audio_encoder_checkpoints(checkpoint, prefix):
    state = torch.load(checkpoint, weights_only=True)["model_state_dict"]
    state = {
        prefix + key.removeprefix("audio_encoder.encoder."): value
        for key, value in state.items()
        if key.startswith("audio_encoder.encoder.")
    }
    torch.save(state, checkpoint)
    VoiceFMContentEncoder(checkpoint, device="cpu")


def test_voicefm_rejects_long_audio_and_invalid_layer(checkpoint):
    encoder = VoiceFMContentEncoder(checkpoint, device="cpu")
    with pytest.raises(ValueError, match="30 seconds"):
        encoder.encode_waveforms(torch.zeros(1, 480001))
    with pytest.raises(ValueError, match="16000"):
        encoder.encode_waveforms(torch.zeros(1, 1000), sample_rate=8000)
    encoder.layer = 5
    with pytest.raises(ValueError, match="Invalid VoiceFM layer"):
        encoder.encode_waveforms(torch.zeros(1, 1000))


def test_voicefm_alias():
    assert resolve_content_encoder("voicefm") is VoiceFMContentEncoder


def test_voicefm_real_whisper_backend_local_roundtrip(tmp_path):
    transformers = pytest.importorskip("transformers")
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    config = transformers.WhisperConfig(d_model=8, encoder_layers=2, encoder_attention_heads=2, encoder_ffn_dim=16)
    config.save_pretrained(tmp_path)
    transformers.WhisperFeatureExtractor().save_pretrained(tmp_path)
    weights = tmp_path / "weights.pt"
    torch.save(
        {"model_state_dict": {"audio_encoder.encoder." + k: v for k, v in WhisperEncoder(config).state_dict().items()}},
        weights,
    )
    encoder = VoiceFMContentEncoder(weights, model_name=str(tmp_path), layer=None, device="cpu", local_files_only=True)
    features = encoder.encode_waveforms(torch.randn(2, 1600), torch.tensor([1600, 1001]))
    assert features.values.shape == (2, 5, 2, 8)
    assert features.lengths.tolist() == [5, 4]
    assert encoder.frame_hz == 50

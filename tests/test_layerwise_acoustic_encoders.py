import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from quick_convert.components.layers import LayerWeightedSum
from quick_convert.components.ssl import EmotionEncoder, PASEContentEncoder


class FakeBlock(nn.Module):
    def forward(self, values):
        state = values + 1
        return state, state + 1000


class FakeEmotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = {"embed_dim": 4, "normalize": True}
        self.blocks = nn.ModuleList([FakeBlock(), FakeBlock()])
        self.modality_encoders = {"AUDIO": SimpleNamespace(modality_cfg=SimpleNamespace(num_extra_tokens=2))}
        self.fail = False
        self.seen = []

    def extract_features(self, source, padding_mask=None, mask=False):
        assert padding_mask is None and mask is False
        self.seen.append(source.clone())
        frames = source.shape[-1] // 320
        values = torch.zeros(1, frames + 2, 4, device=source.device)
        for block in self.blocks:
            values, _target = block(values)
            if self.fail:
                raise RuntimeError("synthetic failure")
        return {"x": values[:, 2:] + 10, "padding_mask": None}


@pytest.fixture
def emotion_model(monkeypatch):
    model = FakeEmotionModel()
    wrapper = SimpleNamespace(model=model)
    monkeypatch.setitem(sys.modules, "funasr", SimpleNamespace(AutoModel=lambda **kwargs: wrapper))
    return model


def test_emotion_all_layers_use_states_remove_tokens_and_normalize(emotion_model):
    encoder = EmotionEncoder(layer=None, device="cpu")
    audio = torch.randn(2, 1600)
    original = audio.clone()
    output = encoder.encode_waveforms(audio, torch.tensor([1600, 960]))
    assert output.values.shape == (2, 5, 2, 4)
    assert output.lengths.tolist() == [5, 3]
    assert encoder.N_LAYERS == 2
    assert torch.all(output.values[0, :, 0] == 1)
    assert torch.all(output.values[0, :, 1] == 12)
    assert torch.all(output.values[1, 3:] == 0)
    assert torch.equal(audio, original)
    torch.testing.assert_close(emotion_model.seen[0][0], nn.functional.layer_norm(audio[0], (1600,)))
    assert all(not block._forward_hooks for block in emotion_model.blocks)
    fusion = LayerWeightedSum(num_layers=2)
    values = fusion(output.values.detach().clone())
    values.square().mean().backward()
    assert fusion.weights.grad is not None
    assert torch.isfinite(fusion.weights.grad).all()


def test_emotion_selected_layer_and_hook_cleanup(emotion_model):
    encoder = EmotionEncoder(layer=0, device="cpu")
    output = encoder.encode_waveforms(torch.randn(1, 1600))
    assert output.values.shape == (1, 5, 4)
    assert torch.all(output.values == 1)
    encoder.layer = 2
    with pytest.raises(ValueError, match="Invalid emotion2vec layer"):
        encoder.encode_waveforms(torch.randn(1, 1600))
    encoder.layer = None
    emotion_model.fail = True
    with pytest.raises(RuntimeError, match="synthetic failure"):
        encoder.encode_waveforms(torch.randn(1, 1600))
    assert all(not block._forward_hooks for block in emotion_model.blocks)


class FakeDensePASE(nn.Module):
    emb_dim = 4
    densemerge = "sum"

    def __init__(self):
        super().__init__()
        self.denseskips = nn.ModuleList([nn.Conv1d(1, 4, 1, stride=80, bias=False)])
        nn.init.constant_(self.denseskips[0].weight, 2)
        self.fail = False

    def fuse_skip(self, final, skip):
        factor = skip.shape[-1] // final.shape[-1]
        skip = skip[:, :, : final.shape[-1] * factor]
        return final + skip.reshape(1, 4, final.shape[-1], factor).mean(dim=-1)

    def forward(self, audio):
        skip = self.denseskips[0](audio)
        if self.fail:
            raise RuntimeError("synthetic failure")
        final = torch.zeros(1, 4, audio.shape[-1] // 160)
        return self.fuse_skip(final, skip) + 10


def test_pase_projected_layers_alignment_selection_and_cleanup(monkeypatch, tmp_path):
    model = FakeDensePASE()
    monkeypatch.setitem(sys.modules, "pase", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "pase.models", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "pase.models.frontend", SimpleNamespace(wf_builder=lambda config: model))
    config = tmp_path / "frontend.json"
    config.write_text('{"strides": [10, 16]}')
    checkpoint = tmp_path / "weights.pt"
    torch.save(model.state_dict(), checkpoint)
    encoder = PASEContentEncoder(config, checkpoint, layer=None, device="cpu")
    audio = torch.arange(1600, dtype=torch.float32).repeat(2, 1)
    output = encoder.encode_waveforms(audio, torch.tensor([1600, 800]))
    assert output.values.shape == (2, 10, 2, 4)
    assert output.lengths.tolist() == [10, 5]
    assert encoder.N_LAYERS == 2
    expected = 2 * torch.arange(0, 1600, 80, dtype=torch.float32).reshape(10, 2).mean(dim=-1)
    torch.testing.assert_close(output.values[0, :, 0, 0], expected)
    torch.testing.assert_close(output.values[0, :, 1, 0], expected + 10)
    encoder.layer = -1
    final = encoder.encode_waveforms(audio, torch.tensor([1600, 800]))
    torch.testing.assert_close(output.values[:, :, -1], final.values)
    encoder.layer = 0
    selected = encoder.encode_waveforms(audio, torch.tensor([1600, 800]))
    torch.testing.assert_close(output.values[:, :, 0], selected.values)
    fusion = LayerWeightedSum(num_layers=2)
    fusion(output.values.detach().clone()).square().mean().backward()
    assert fusion.weights.grad is not None
    assert torch.isfinite(fusion.weights.grad).all()
    encoder.layer = None
    model.fail = True
    with pytest.raises(RuntimeError, match="synthetic failure"):
        encoder.encode_waveforms(audio)
    assert all(not projection._forward_hooks for projection in model.denseskips)

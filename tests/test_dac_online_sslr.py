"""Online DAC composition, valid-frame isolation and lazy optional dependencies."""

from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from quick_convert.components.ssl import DACContentEncoder


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 4, 3, stride=2, padding=1)
        self.sizes = []

    def forward(self, x):
        self.sizes.append(x.shape)
        return self.conv(x)


def encoder():
    model = DACContentEncoder.__new__(DACContentEncoder)
    nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model._sample_rate = 16000
    model.hop_length = 2
    model.FEATURE_DIM = 4
    model.model_name = "dac::test"
    model.dac_encoder = TinyEncoder()
    return model


def test_dac_padding_isolation_groups_equal_hops_and_preserves_storage():
    model = encoder()
    values = torch.randn(3, 10, requires_grad=True)
    lengths = torch.tensor([10, 5, 6])
    original = values.detach().clone()
    output = model.encode_waveforms(values, lengths)
    assert output.values.shape == (3, 5, 1, 4)
    assert output.lengths.tolist() == [5, 3, 3]
    assert len(model.dac_encoder.sizes) == 2
    assert model.dac_encoder.sizes[-1] == (2, 1, 6)
    # Tail junk and another recording's length must not affect valid features.
    individual = model.encode_waveforms(values[1:2, :5], torch.tensor([5]))
    torch.testing.assert_close(output.values[1, :3], individual.values[0])
    assert output.values[1, 3:].count_nonzero() == 0
    output.values.sum().backward()
    assert values.grad[1, 5:].count_nonzero() == 0
    assert values.grad[2, 6:].count_nonzero() == 0
    torch.testing.assert_close(values, original)


def test_dac_tracks_model_device_instead_of_original_preference():
    model = encoder()
    model.device = torch.device("meta")
    result = model.encode_waveforms(torch.ones(1, 6), torch.tensor([6]))
    assert result.values.device.type == "cpu"


@pytest.mark.parametrize("lengths", [torch.tensor([0]), torch.tensor([7]), torch.tensor([3.0])])
def test_dac_invalid_lengths(lengths):
    with pytest.raises(ValueError):
        encoder().encode_waveforms(torch.ones(1, 6), lengths)


def test_dac_default_capabilities_match_pretrained_checkpoint(monkeypatch):
    module = ModuleType("dac")
    module.utils = SimpleNamespace(download=lambda **kw: "weights")
    module.DAC = SimpleNamespace(
        load=lambda path: SimpleNamespace(encoder=TinyEncoder(), sample_rate=16000, hop_length=320, latent_dim=1024)
    )
    monkeypatch.setitem(__import__("sys").modules, "dac", module)
    model = DACContentEncoder(device="cpu")
    assert model.feature_dim == model.FEATURE_DIM == 1024
    assert model.frame_hz == model.FRAME_HZ == 50
    assert model.sample_rate == model.SAMPLE_RATE == 16000
    assert not any(p.requires_grad for p in model.parameters())
    with pytest.raises(ValueError, match="sample_rate"):
        DACContentEncoder(device="cpu", sample_rate=24000)


@pytest.mark.parametrize("backbone", ["cmdiff", "dit"])
def test_online_dac_campp_run_composes_and_selects_required_profiles(backbone):
    from hydra import compose, initialize_config_dir

    from quick_convert.utils.resolvers import register_config_resolvers

    register_config_resolvers()
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / "quick_convert/configs")
    ):
        cfg = compose(config_name=f"run/train_sslr_dac_{backbone}_campp_prefix")
        assert cfg.system.online_encoders.content._target_.endswith("DACContentEncoder")
        assert cfg.system.feature_transform.num_layers == 1
        assert cfg.system.decoder.feature_dim == 1024
        assert cfg.system.decoder.flow.input_frame_rate == 50
        assert cfg.system.decoder.flow.token_mel_ratio == 1
        assert cfg.system.decoder.flow.decoder.estimator.spk_dim is None
        assert cfg.system.online_encoders.speaker is None
        assert cfg.train_dataset.target_sr == 16000


@pytest.mark.parametrize("backbone", ["cmdiff", "dit"])
def test_online_dac_run_dependencies_do_not_include_transformers(backbone):
    from quick_convert.dependencies import compose_dependency_plan

    plan = compose_dependency_plan(f"train_sslr_dac_{backbone}_campp_prefix")
    assert {"dac", "campplus-frames", "cosyvoice", "training"} <= set(plan.extras)
    assert "transformers" not in plan.extras

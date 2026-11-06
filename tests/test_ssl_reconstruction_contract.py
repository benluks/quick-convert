from pathlib import Path

import pytest
import torch
from torch import nn

from quick_convert.components.decoders import CosyVoiceDecoderOutput
from quick_convert.components.layers import LayerWeightedSum
from quick_convert.components.layers.rvq import RVQLosses, RVQOutput
from quick_convert.components.mixins.resource import ResolvedResource
from quick_convert.data import AudioBatch
from quick_convert.systems.conversion import SSLReconstructionConversionSystem


pytest.importorskip("lightning")

from quick_convert.pipelines.training.modules.ssl_reconstruction import (
    SSLReconstructionTrainingModule as LegacySSLReconstructionTrainingModule,
)
from quick_convert.training.lightning.modules.ssl_reconstruction import (
    SSLReconstructionTrainingModule,
)
from quick_convert.training.lightning.optim import Optimization


def test_legacy_training_module_import_is_compatible():
    assert LegacySSLReconstructionTrainingModule is SSLReconstructionTrainingModule


class FakeRVQEncoder(nn.Module):
    @staticmethod
    def output_lengths(input_lengths):
        return input_lengths

    def forward(self, features, padding_mask):
        assert padding_mask.tolist() == [[True, True, True], [True, True, False]]
        loss = features.new_tensor(0.25)
        return RVQOutput(
            z_q=features + 10,
            layer_z_qs=[features + 10],
            codes=torch.zeros(features.shape[0], 1, features.shape[2], dtype=torch.long),
            latents=features - 10,
            loss=RVQLosses(loss=loss, raw={"commitment": loss}, weighted={"commitment": loss}),
        )


class FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.flow = nn.Module()
        self.loss_calls = 0

    def compute_loss(self, **kwargs):
        self.loss_calls += 1
        assert kwargs["features"].shape == (2, 3, 4)
        assert kwargs["lengths"].tolist() == [3, 2]
        return CosyVoiceDecoderOutput(loss=kwargs["features"].new_tensor(0.5))


def test_ssl_reconstruction_training_module_accepts_an_explicit_system():
    system = SSLReconstructionConversionSystem(
        decoder=FakeDecoder(),
        feature_transform=LayerWeightedSum(num_layers=2),
        encoder=FakeRVQEncoder(),
    )

    module = SSLReconstructionTrainingModule(
        system=system,
        optimization=Optimization(lr_scheduler=None),
    )

    assert module.system is system


def test_ssl_reconstruction_maps_legacy_checkpoint_keys():
    module = SSLReconstructionTrainingModule(
        decoder=FakeDecoder(),
        feature_transform=LayerWeightedSum(num_layers=2),
        encoder=FakeRVQEncoder(),
        optimization=Optimization(lr_scheduler=None),
    )
    current_state = module.state_dict()
    legacy_state = {key.removeprefix("system."): value.clone() for key, value in current_state.items()}

    prepared_state = module._prepare_checkpoint_state_dict(legacy_state)

    assert prepared_state.keys() == current_state.keys()
    module.load_state_dict(prepared_state, strict=True)


def test_ssl_reconstruction_training_step_uses_the_wrapped_system(monkeypatch):
    module = SSLReconstructionTrainingModule(
        decoder=FakeDecoder(),
        feature_transform=LayerWeightedSum(num_layers=2),
        encoder=FakeRVQEncoder(),
        optimization=Optimization(lr_scheduler=None),
    )
    monkeypatch.setattr(module, "log_dict", lambda *args, **kwargs: None)
    batch = AudioBatch(
        utt_ids=["a", "b"],
        paths=[Path("a.wav"), Path("b.wav")],
        splits=[None, None],
        resources={
            "content": ResolvedResource(
                values=torch.ones(2, 3, 2, 4),
                lengths=torch.tensor([3, 2]),
            ),
            "speaker": torch.ones(2, 5),
        },
        waveforms=torch.ones(2, 160),
        lengths=torch.tensor([160, 120]),
        sample_rates=torch.tensor([16_000, 16_000]),
    )

    output = module._shared_step(batch, "val")

    assert module.decoder.loss_calls == 1
    torch.testing.assert_close(output.features, torch.full((2, 3, 4), 11.0))
    torch.testing.assert_close(output.lengths, batch.resources["content"].lengths)
    assert output.encoder_output is not None
    assert not torch.equal(output.features, output.encoder_output.latents.transpose(1, 2))
    torch.testing.assert_close(output.loss, torch.tensor(0.75))


def test_ssl_training_keeps_teachers_and_frozen_vocoder_in_eval():
    decoder = FakeDecoder()
    decoder.vocoder = nn.Sequential(nn.BatchNorm1d(4), nn.Dropout()).requires_grad_(False)
    system = SSLReconstructionConversionSystem(
        decoder=decoder,
        online_encoders={"content": nn.Dropout()},
        feature_transform=LayerWeightedSum(2),
    )
    module = SSLReconstructionTrainingModule(system=system, optimization=Optimization())
    module.eval()
    module.train()
    assert module.feature_transform.training
    assert module.decoder.training
    assert not module.online_encoders["content"].training
    assert not decoder.vocoder.training
    assert not any(p.requires_grad for p in module.online_encoders.parameters())


def test_ssl_gradient_groups_include_fusion_and_input_projection():
    decoder = FakeDecoder()
    decoder.input_projection = nn.Linear(4, 4)
    module = SSLReconstructionTrainingModule(
        decoder=decoder,
        feature_transform=LayerWeightedSum(2),
        optimization=Optimization(),
        enable_grad_norm_logging=False,
    )
    assert module.grad_norm_modules["feature_transform"] is module.feature_transform
    assert module.grad_norm_modules["input_projection"] is decoder.input_projection
    assert not module.enable_grad_norm_logging


def test_ssl_compile_replaces_flow_used_by_training():
    from quick_convert.training.lightning.trainer import LightningTrainer

    decoder = FakeDecoder()
    decoder.flow = nn.Linear(4, 4)
    module = SSLReconstructionTrainingModule(decoder=decoder, optimization=Optimization())
    original_keys = set(module.state_dict())
    compilations = []

    def backend(graph, inputs):
        compilations.append(graph)
        return graph.forward

    trainer = LightningTrainer(
        module, compile={"enabled": True, "targets": ["system.decoder.flow"], "backend": backend}
    )
    trainer._maybe_compile_module()
    x = torch.randn(2, 4, requires_grad=True)
    module.decoder.flow(x).sum().backward()
    assert compilations
    assert torch.isfinite(x.grad).all()
    assert "decoder" not in module._modules
    assert set(module._clean_compiled_state_dict(module.state_dict())) == original_keys


def test_ssl_representation_logging_masks_padding_and_logs_layer_weights(monkeypatch):
    from types import SimpleNamespace

    encoder = FakeRVQEncoder()
    encoder.codebook_size = 4
    encoder.n_codebooks = 2
    module = SSLReconstructionTrainingModule(
        decoder=FakeDecoder(),
        feature_transform=LayerWeightedSum(2),
        encoder=encoder,
        optimization=Optimization(),
    )
    bars = []
    metrics = {}
    module._media_logger = SimpleNamespace(log_bar=lambda *args, **kwargs: bars.append(args))
    monkeypatch.setattr(module, "log_dict", lambda values, **kwargs: metrics.update(values))
    output = SimpleNamespace(codes=torch.tensor([[[0, 1, 3]], [[1, 3, 3]]]))
    module._log_representation_state(output, torch.tensor([2, 1]))
    assert len(bars) == 1
    torch.testing.assert_close(bars[0][1], torch.tensor([0.5, 0.5]))
    assert metrics["rvq/codebook_0/utilization"] == 0.5
    torch.testing.assert_close(metrics["rvq/codebook_0/max_usage"], torch.tensor(2 / 3))
    assert metrics["rvq/codebook_1/perplexity"] == 0
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def reduce_counts(counts):
        assert counts.shape == (2, 4)
        # Another rank used the codebook absent from this rank's output.
        counts[1, 2] += 3

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce_counts)
    module._log_representation_state(output, torch.tensor([2, 1]))
    assert metrics["rvq/codebook_1/perplexity"] == 1
    assert metrics["rvq/codebook_1/utilization"] == 0.25
    module._log_representation_state(None, torch.tensor([2, 1]))


@pytest.mark.parametrize("slot", ["current", "legacy"])
def test_from_run_migrates_old_system_targets_and_parameter_keys(tmp_path, slot):
    from omegaconf import OmegaConf

    module_config = {
        "_target_": "quick_convert.training.lightning.modules.ssl_reconstruction.SSLReconstructionTrainingModule",
        "optimization": {"_target_": "quick_convert.training.lightning.optim.Optimization", "lr_scheduler": None},
        "system": {
            "_target_": "quick_convert.systems.reconstruction.ssl.SSLReconstructionSystem",
            "decoder": {"_target_": "test_ssl_reconstruction_contract.FakeDecoder"},
            "feature_transform": {"_target_": "quick_convert.components.layers.LayerWeightedSum", "num_layers": 2},
        },
    }
    expected = SSLReconstructionTrainingModule(
        optimization=Optimization(lr_scheduler=None),
        system=SSLReconstructionConversionSystem(
            decoder=FakeDecoder(), feature_transform=LayerWeightedSum(num_layers=2)
        ),
    )
    with torch.no_grad():
        for parameter in expected.parameters():
            parameter.fill_(0.73)
    config = {"trainer": {"module": module_config}}
    if slot == "legacy":
        config = {"pipeline": config}
    OmegaConf.save(OmegaConf.create(config), tmp_path / "config.yaml")
    (tmp_path / "checkpoints").mkdir()
    torch.save(
        {"state_dict": {k.removeprefix("system."): v for k, v in expected.state_dict().items()}},
        tmp_path / "checkpoints/last.ckpt",
    )
    restored = SSLReconstructionTrainingModule.from_run(tmp_path, map_location="cpu")
    assert set(restored.state_dict()) == set(expected.state_dict())
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)

from pathlib import Path

import pytest
import torch
from torch import nn

from quick_convert.components.decoders import CosyVoiceDecoderOutput
from quick_convert.components.layers import LayerWeightedSum
from quick_convert.components.layers.rvq import RVQLosses, RVQOutput
from quick_convert.components.mixins.resource import ResolvedResource
from quick_convert.data import AudioBatch
from quick_convert.systems.reconstruction import SSLReconstructionSystem


pytest.importorskip("lightning")

from quick_convert.pipelines.training.modules.ssl_reconstruction import (  # noqa: E402
    SSLReconstructionTrainingModule,
)
from quick_convert.pipelines.training.optim.base import Optimization


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


def test_ssl_reconstruction_encoder_uses_quantized_output_and_preserves_lengths():
    module = SSLReconstructionTrainingModule(
        decoder=FakeDecoder(),
        feature_transform=LayerWeightedSum(num_layers=2),
        encoder=FakeRVQEncoder(),
        optimization=Optimization(lr_scheduler=None),
    )
    content = ResolvedResource(
        values=torch.ones(2, 3, 2, 4),
        lengths=torch.tensor([3, 2]),
    )

    features, lengths, encoder_output = module._encode_content(content)

    assert torch.equal(features, torch.full((2, 3, 4), 11.0))
    assert torch.equal(lengths, content.lengths)
    assert encoder_output is not None
    assert not torch.equal(features, encoder_output.latents.transpose(1, 2))


def test_ssl_reconstruction_training_module_accepts_an_explicit_system():
    system = SSLReconstructionSystem(
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
    torch.testing.assert_close(output.loss, torch.tensor(0.75))

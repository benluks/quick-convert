import pytest
import torch
from torch import nn

from quick_convert.components.layers import LayerWeightedSum
from quick_convert.components.layers.rvq import RVQLosses, RVQOutput
from quick_convert.components.mixins.resource import ResolvedResource


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

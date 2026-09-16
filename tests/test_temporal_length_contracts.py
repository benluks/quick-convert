import pytest
import torch
from torch import nn

from quick_convert.components.encoders import ConformerEncoder, ConformerEncoderSSL, ParallelConformerEncoder
from quick_convert.components.layers import LayerWeightedSum
from quick_convert.components.layers.rvq_ema import ResidualVectorQuantizerEMA


@pytest.mark.parametrize(
    "module_type",
    [
        LayerWeightedSum,
        ConformerEncoder,
        ConformerEncoderSSL,
        ParallelConformerEncoder,
        ResidualVectorQuantizerEMA,
    ],
)
def test_time_preserving_modules_report_unchanged_lengths(module_type):
    lengths = torch.tensor([12, 7])

    assert torch.equal(module_type.output_lengths(lengths), lengths)


def test_layer_weighted_sum_preserves_time_and_projects_features():
    fusion = LayerWeightedSum(num_layers=3, projection=nn.Linear(4, 2))

    output = fusion(torch.zeros(2, 5, 3, 4))

    assert output.shape == (2, 5, 2)


def test_layer_weighted_sum_rejects_wrong_layer_count():
    fusion = LayerWeightedSum(num_layers=3)

    with pytest.raises(ValueError, match="Expected 3 representation layers"):
        fusion(torch.zeros(2, 5, 2, 4))

import torch

from quick_convert.components.layers.rvq import RVQLosses, RVQOutput
from quick_convert.components.layers.rvq_ema import ResidualVectorQuantizerEMA


def test_rvq_ema_implements_shared_output_contract():
    quantizer = ResidualVectorQuantizerEMA(
        input_dim=4,
        n_codebooks=2,
        codebook_size=8,
        codebook_dim=2,
        kmeans_init=False,
        threshold_ema_dead_code=0,
        loss_weights={"commitment": 0.25},
    ).eval()

    values = torch.randn(2, 4, 5)
    valid = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
        ]
    )

    output = quantizer(values, valid)

    assert isinstance(output, RVQOutput)
    assert isinstance(output.loss, RVQLosses)
    assert output.z_q.shape == (2, 4, 5)
    assert len(output.layer_z_qs) == 2
    assert all(layer.shape == (2, 4, 5) for layer in output.layer_z_qs)
    assert output.codes.shape == (2, 2, 5)
    assert output.latents.shape == (2, 4, 5)
    assert output.loss.loss.ndim == 0
    assert output.loss.raw.keys() == output.loss.weighted.keys()


def test_rvq_layer_contributions_sum_to_quantized_output():
    quantizer = ResidualVectorQuantizerEMA(
        input_dim=4,
        n_codebooks=2,
        codebook_size=8,
        codebook_dim=2,
        kmeans_init=False,
        threshold_ema_dead_code=0,
    ).eval()

    values = torch.randn(2, 4, 5)
    valid = torch.ones(2, 5, dtype=torch.bool)

    output = quantizer(values, valid)

    torch.testing.assert_close(sum(output.layer_z_qs), output.z_q)

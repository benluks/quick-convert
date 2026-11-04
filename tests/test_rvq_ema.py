import pytest
import torch

from quick_convert.components.layers.rvq import RVQLosses, RVQOutput
from quick_convert.components.layers.rvq_ema import ResidualVectorQuantizerEMA
from quick_convert.components.layers.vector_quantize import VectorQuantize


@pytest.mark.parametrize("topk", [None, 1])
@pytest.mark.parametrize("use_cosine_sim", [False, True])
def test_commitment_uses_quantizer_space_with_or_without_mask(topk, use_cosine_sim):
    quantizer = VectorQuantize(
        dim=3,
        codebook_dim=2,
        codebook_size=2,
        use_cosine_sim=use_cosine_sim,
        kmeans_init=False,
        ema_update=False,
        rotation_trick=False,
    ).train()
    with torch.no_grad():
        quantizer.project_in.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        quantizer.project_in.bias.zero_()
        quantizer.codebook = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    values = torch.tensor([[[3.0, 0.0, 7.0], [0.0, 4.0, 8.0]]], requires_grad=True)
    losses = []
    gradients = []
    for mask in (None, torch.ones(1, 2, dtype=torch.bool)):
        _, _, _, breakdown = quantizer(values, mask=mask, topk=topk, freeze_codebook=True, return_loss_breakdown=True)
        losses.append(breakdown.commitment)
        gradients.append(torch.autograd.grad(breakdown.commitment.sum(), values)[0])

    expected = 0.0 if use_cosine_sim else 3.25
    for loss in losses:
        torch.testing.assert_close(loss.mean(), torch.tensor(expected))
    torch.testing.assert_close(gradients[0], gradients[1])


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

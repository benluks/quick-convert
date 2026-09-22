import torch

from quick_convert.components.ssl.s3tokenizer import S3TokenizerContentEncoder


def test_s3_ternary_factors_match_fsq_rounding():
    values = torch.tensor([[[-1.0, -0.6, -0.49, 0.0, 0.49, 0.6, 0.99, 1.0]]])

    factors = S3TokenizerContentEncoder._ternary_factors(values)

    assert factors.dtype == torch.int64
    assert factors.tolist() == [[[-1, -1, 0, 0, 0, 1, 1, 1]]]


def test_s3_ternary_packing_matches_base_three_codebook():
    factors = torch.tensor(
        [
            [[-1] * 8],
            [[0] * 8],
            [[1] * 8],
        ],
        dtype=torch.int64,
    )

    tokens = S3TokenizerContentEncoder._pack_ternary(factors)

    assert tokens.tolist() == [[0], [3280], [6560]]


def test_s3_declares_continuous_and_discrete_representations():
    assert S3TokenizerContentEncoder.REPRESENTATIONS == (
        "encoder",
        "pre_tanh",
        "post_tanh",
        "ternary",
        "tokens",
    )
    assert S3TokenizerContentEncoder.N_LAYERS == 12
    assert S3TokenizerContentEncoder.FEATURE_DIM == 1280
    assert S3TokenizerContentEncoder.BOTTLENECK_DIM == 8

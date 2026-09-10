import pytest
import torch

from quick_convert.components.feature_extractors.base import BaseFeatureExtractor
from quick_convert.components.ssl.base import ContentFeatures
from quick_convert.components.ssl.dac import DACContentEncoder
from quick_convert.components.ssl.wavlm import WavLMContentEncoder


def test_feature_extractor_requires_a_feature_name():
    class MissingFeatureName(BaseFeatureExtractor):
        pass

    with pytest.raises(TypeError, match="feature_name"):
        MissingFeatureName()


def make_content_features(**overrides):
    fields = {
        "values": torch.zeros(2, 5, 4),
        "lengths": torch.tensor([5, 3]),
        "feature_dim": 4,
        "representation_type": "continuous",
        "temporal_granularity": "frame",
        "backend": "test",
        "model_name": "test-model",
        "layer": None,
    }
    fields.update(overrides)
    return ContentFeatures(**fields)


def test_content_features_accept_padded_values_and_valid_lengths():
    features = make_content_features()

    assert features.values.shape == (2, 5, 4)
    assert features.lengths.tolist() == [5, 3]


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"values": [torch.zeros(5, 4)]}, TypeError, "values must be a tensor"),
        ({"lengths": [5, 3]}, TypeError, "lengths must be a tensor"),
        ({"lengths": torch.tensor([6, 3])}, ValueError, "exceeds the padded time dimension"),
        ({"feature_dim": 3}, ValueError, "does not match"),
    ],
)
def test_content_features_reject_invalid_batch_contracts(overrides, error, message):
    with pytest.raises(error, match=message):
        make_content_features(**overrides)


def test_wavlm_reports_exact_convolutional_output_lengths():
    encoder = WavLMContentEncoder.__new__(WavLMContentEncoder)
    encoder.model = type(
        "FakeModel",
        (),
        {"config": type("FakeConfig", (), {"conv_kernel": (10, 3), "conv_stride": (5, 2)})()},
    )()

    assert encoder.output_lengths(torch.tensor([100, 50])).tolist() == [9, 4]


def test_dac_reports_exact_hop_output_lengths():
    encoder = DACContentEncoder.__new__(DACContentEncoder)
    encoder.hop_length = 320

    assert encoder.output_lengths(torch.tensor([1, 320, 321, 640])).tolist() == [1, 1, 2, 2]

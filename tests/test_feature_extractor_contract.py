import inspect

import pytest
import torch

from quick_convert.components.feature_extractors.base import BaseFeatureExtractor
from quick_convert.components.ssl import (
    DACContentEncoder,
    EmotionEncoder,
    ProsodyEncoder,
    S3TokenizerContentEncoder,
    W2VBertContentEncoder,
    WavLMContentEncoder,
)
from quick_convert.components.ssl.base import ContentEncoder, ContentFeatures


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


@pytest.mark.parametrize(
    "encoder_class",
    [
        DACContentEncoder,
        EmotionEncoder,
        ProsodyEncoder,
        S3TokenizerContentEncoder,
        W2VBertContentEncoder,
        WavLMContentEncoder,
    ],
)
def test_built_in_content_encoders_implement_the_complete_contract(encoder_class):
    assert not inspect.isabstract(encoder_class)


def test_content_encoder_contract_requires_batch_and_timebase_interfaces():
    assert {"forward", "sample_rate", "frame_hz"} <= ContentEncoder.__abstractmethods__


def test_fixed_rate_encoders_expose_timebase_without_loading_models():
    assert S3TokenizerContentEncoder.__new__(S3TokenizerContentEncoder).frame_hz == 25.0

    w2vbert = W2VBertContentEncoder.__new__(W2VBertContentEncoder)
    w2vbert.processor = type("FakeProcessor", (), {"stride": 2})()
    assert w2vbert.frame_hz == 50.0


def test_wavlm_derives_timebase_from_the_loaded_frontend():
    encoder = WavLMContentEncoder.__new__(WavLMContentEncoder)
    encoder._sample_rate = 16_000
    encoder.model = type(
        "FakeModel", (), {"config": type("FakeConfig", (), {"conv_stride": (5, 2, 2, 2, 2, 2, 2)})()}
    )()

    assert encoder.frame_hz == 50.0


def test_utterance_representation_has_no_frame_rate():
    encoder = EmotionEncoder.__new__(EmotionEncoder)
    encoder.granularity = "utterance"

    assert encoder.frame_hz is None

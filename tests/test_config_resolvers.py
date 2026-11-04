import pytest

from quick_convert.components.ssl import W2VBertContentEncoder
from quick_convert.utils.resolvers import class_attribute, frame_ratio


def test_class_attribute_reads_public_capability_constant():
    target = "quick_convert.components.ssl.W2VBertContentEncoder"

    assert class_attribute(target, "SAMPLE_RATE") == W2VBertContentEncoder.SAMPLE_RATE


@pytest.mark.parametrize("attribute", ["sample_rate", "_PRIVATE"])
def test_class_attribute_rejects_non_public_constants(attribute):
    with pytest.raises(ValueError, match="public uppercase"):
        class_attribute("quick_convert.components.ssl.W2VBertContentEncoder", attribute)


@pytest.mark.parametrize("input_hz,expected", [(50, 1), (25, 2), (12.5, 4)])
def test_frame_ratio(input_hz, expected):
    assert frame_ratio(50, input_hz) == expected


@pytest.mark.parametrize("input_hz", [0, -25, 30, 100, float("nan"), float("inf")])
def test_frame_ratio_rejects_unsupported_rates(input_hz):
    with pytest.raises(ValueError):
        frame_ratio(50, input_hz)

import pytest

from quick_convert.components.ssl import W2VBertContentEncoder
from quick_convert.utils.resolvers import class_attribute


def test_class_attribute_reads_public_capability_constant():
    target = "quick_convert.components.ssl.W2VBertContentEncoder"

    assert class_attribute(target, "SAMPLE_RATE") == W2VBertContentEncoder.SAMPLE_RATE


@pytest.mark.parametrize("attribute", ["sample_rate", "_PRIVATE"])
def test_class_attribute_rejects_non_public_constants(attribute):
    with pytest.raises(ValueError, match="public uppercase"):
        class_attribute("quick_convert.components.ssl.W2VBertContentEncoder", attribute)

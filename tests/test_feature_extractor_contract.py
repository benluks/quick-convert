import pytest

from quick_convert.components.feature_extractors.base import BaseFeatureExtractor


def test_feature_extractor_requires_a_feature_name():
    class MissingFeatureName(BaseFeatureExtractor):
        pass

    with pytest.raises(TypeError, match="feature_name"):
        MissingFeatureName()

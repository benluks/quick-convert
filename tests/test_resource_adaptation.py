from types import SimpleNamespace

import pytest
import torch

from quick_convert.components.mixins.resource import OnlineResourceMixin, ResolvedResource
from quick_convert.components.ssl.base import ContentFeatures
from quick_convert.data.resources.base import TensorResourceBatch


def test_content_metadata_survives_resolution_and_repeated_adaptation():
    content = ContentFeatures(
        values=torch.ones(2, 3, 4, requires_grad=True),
        lengths=torch.tensor([3, 2]),
        feature_dim=4,
        representation_type="continuous",
        temporal_granularity="frame",
        backend="test",
        model_name="tiny",
        layer=6,
        frame_hz=50.0,
    )

    resolved = OnlineResourceMixin._normalize_resource(content)
    adapted = OnlineResourceMixin._normalize_resource(resolved)

    assert adapted.metadata == {
        "feature_dim": 4,
        "representation_type": "continuous",
        "temporal_granularity": "frame",
        "backend": "test",
        "model_name": "tiny",
        "layer": 6,
        "frame_hz": 50.0,
    }
    torch.testing.assert_close(adapted.values, content.values)
    assert adapted.lengths is content.lengths
    assert not adapted.values.requires_grad
    assert content.values.requires_grad
    adapted.metadata["layer"] = 9
    assert resolved.metadata["layer"] == content.layer == 6


@pytest.mark.parametrize(
    "wrapper", [TensorResourceBatch, ResolvedResource, SimpleNamespace, lambda **kw: tuple(kw.values())]
)
def test_precomputed_and_legacy_wrappers_keep_values_and_lengths_without_inventing_metadata(wrapper):
    values = torch.ones(2, 3, 4, requires_grad=True)
    lengths = torch.tensor([3, 2])

    resolved = OnlineResourceMixin._normalize_resource(wrapper(values=values, lengths=lengths))

    torch.testing.assert_close(resolved.values, values)
    assert resolved.lengths is lengths
    assert resolved.metadata is None
    assert not resolved.values.requires_grad


@pytest.mark.parametrize("value", [torch.ones(2, 4), {"transcript": ["hello", "world"]}, ["hello", "world"]])
def test_plain_resources_remain_values(value):
    resolved = OnlineResourceMixin._normalize_resource(value)

    if isinstance(value, torch.Tensor):
        torch.testing.assert_close(resolved.values, value)
    else:
        assert resolved.values is value
    assert resolved.lengths is None
    assert resolved.metadata is None

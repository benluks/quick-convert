import pytest
import torch

from quick_convert.components.transforms.aligned_crop import aligned_crop


def test_aligned_crop():
    features = torch.arange(10).reshape(1, 10, 1)
    audio = torch.arange(30).reshape(1, 30)
    x, y = aligned_crop(features, [8], audio, [24], hop_size=3, frames=4)
    assert x.shape == (1, 4, 1)
    assert y.shape == (1, 12)
    assert torch.equal(x[0, :, 0] * 3, y[0, ::3])


def test_short_crop_rejected():
    with pytest.raises(ValueError, match="shorter"):
        aligned_crop(torch.zeros(1, 3, 2), [3], torch.zeros(1, 9), [9], hop_size=3, frames=4)

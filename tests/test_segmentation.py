import pytest
import soundfile as sf
import torch

from quick_convert.data.base_dataset import BaseDataset
from quick_convert.data.segmentation import FixedDurationSegmentation
from quick_convert.data.types import MetadataSample


def test_fixed_segmentation(tmp_path):
    path = tmp_path / "audio.wav"
    sf.write(path, torch.linspace(-0.5, 0.5, 40000).numpy(), 16000)
    row = MetadataSample("original", path, split="train", start=0.25, end=2.5)
    dataset = BaseDataset(rows=[row], load=["audio"], segmentation=FixedDurationSegmentation(1))
    assert [(r.start, r.end) for r in dataset.rows] == [(0.25, 1.25), (1.25, 2.25), (2.25, 2.5)]
    assert len({r.utt_id for r in dataset.rows}) == 3
    samples = [dataset[i] for i in range(len(dataset))]
    assert [s.waveform.shape[-1] for s in samples] == [16000, 16000, 4000]
    batch = dataset.collate_fn(samples)
    assert batch[0].parent_utt_id == "original"
    assert batch[0].split == "train"
    original = BaseDataset(rows=[row], load=["audio"])[0]
    torch.testing.assert_close(torch.cat([s.waveform for s in samples], dim=-1), original.waveform)
    assert len(FixedDurationSegmentation(1, drop_remainder=True).build([row])) == 2


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_invalid_duration(duration):
    with pytest.raises(ValueError):
        FixedDurationSegmentation(duration)

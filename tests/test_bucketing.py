import pytest
import torch

from quick_convert.data.bucketing import BucketedBatchSampler, collate_bucket
from quick_convert.data.types import AudioSample


def test_bucket_sampler():
    lengths = [1.0] * 20 + [2.0] * 20 + [3.0] * 20
    sampler = BucketedBatchSampler(lengths, 8, boundaries=[1, 2, 3], max_batch_seconds=12)
    batches = list(sampler)
    assert len(batches) == len(sampler)
    assert sorted(i for batch in batches for i in batch) == list(range(len(lengths)))
    for batch in batches:
        assert len({lengths[i] for i in batch}) == 1
        assert len(batch) * sampler.padded_duration(batch) <= 12
    assert list(sampler) == batches
    sampler.set_epoch(1)
    assert list(sampler) != batches


def test_bucket_overflow():
    with pytest.raises(ValueError, match="largest bucket"):
        BucketedBatchSampler([4], 8, boundaries=[1, 2, 3])


def test_bucket_padding(tmp_path):
    samples = [
        AudioSample(str(i), tmp_path / "audio.wav", waveform=torch.ones(n), sample_rate=16000)
        for i, n in enumerate([17000, 21000])
    ]
    batch = collate_bucket(samples, boundaries=(16000, 32000), sample_rate=16000)
    assert batch.waveforms.shape == (2, 32000)
    assert batch.lengths.tolist() == [17000, 21000]

"""Discrete duration buckets with epoch-local shuffling."""

import math
from bisect import bisect_left
from functools import partial

import torch

from .durations import resolve_durations
from .types import AudioBatch


class BucketedBatchSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        lengths,
        batch_size,
        *,
        boundaries,
        batch_sizes=None,
        max_batch_seconds=None,
        max_batch_attention_cost=None,
        shuffle=True,
        drop_last=False,
        seed=0,
    ):
        self.boundaries = tuple(float(x) for x in boundaries)
        if (
            not self.boundaries
            or any(not math.isfinite(x) or x <= 0 for x in self.boundaries)
            or tuple(sorted(set(self.boundaries))) != self.boundaries
        ):
            raise ValueError("Bucket boundaries must be positive and strictly increasing.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        budgets = [(1, max_batch_seconds), (2, max_batch_attention_cost)]
        for _, budget in budgets:
            if budget is not None and (not math.isfinite(budget) or budget <= 0):
                raise ValueError("Budgets must be finite and positive.")
        if batch_sizes is not None and len(batch_sizes) != len(self.boundaries):
            raise ValueError("Specify one batch size per bucket.")
        self.sizes = []
        for i, boundary in enumerate(self.boundaries):
            limit = min(
                [batch_size]
                + [math.floor(budget / boundary**power) for power, budget in budgets if budget is not None]
            )
            size = limit if batch_sizes is None else batch_sizes[i]
            if not isinstance(size, int) or not 1 <= size <= limit:
                raise ValueError("Bucket batch size must be positive and within budgets.")
            self.sizes.append(size)
        self.groups = [[] for _ in self.boundaries]
        self.assignments = []
        for i, length in enumerate(lengths):
            if not math.isfinite(length) or length <= 0:
                raise ValueError("Lengths must be finite and positive.")
            bucket = bisect_left(self.boundaries, length - 1e-9)
            if bucket == len(self.boundaries):
                raise ValueError("Sample exceeds largest bucket.")
            self.groups[bucket].append(i)
            self.assignments.append(bucket)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    @property
    def sampler(self):
        return self

    def set_epoch(self, epoch):
        self.epoch = epoch

    def padded_duration(self, batch):
        return self.boundaries[self.assignments[batch[0]]]

    def __len__(self):
        return sum(
            len(group) // size if self.drop_last else math.ceil(len(group) / size)
            for group, size in zip(self.groups, self.sizes, strict=True)
        )

    def __iter__(self):
        rng = torch.Generator().manual_seed(self.seed + self.epoch)
        batches = []
        for group, size in zip(self.groups, self.sizes, strict=True):
            indices = [group[i] for i in torch.randperm(len(group), generator=rng).tolist()] if self.shuffle else group
            for start in range(0, len(indices), size):
                batch = indices[start : start + size]
                if len(batch) == size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            batches = [batches[i] for i in torch.randperm(len(batches), generator=rng).tolist()]
        return iter(batches)


def collate_bucket(samples, *, boundaries, sample_rate):
    lengths = [sample.waveform.shape[-1] for sample in samples]
    if any(sample.sample_rate != sample_rate for sample in samples):
        raise ValueError("Bucketed collation requires a common sample rate.")
    bucket = bisect_left(boundaries, max(lengths))
    if bucket == len(boundaries):
        raise ValueError("Loaded waveform exceeds largest bucket.")
    return AudioBatch.from_samples(samples, max_length=boundaries[bucket])


class BucketedSampling:
    def __init__(self, boundaries, duration_cache=None, **options):
        self.boundaries = tuple(boundaries)
        self.duration_cache = duration_cache
        self.options = options

    def build(self, dataset, *, batch_size, shuffle, drop_last):
        if dataset.max_length is not None or not dataset.target_sr:
            raise ValueError("Bucketed sampling requires target_sr and no fixed max_length.")
        if "audio" not in dataset.load:
            raise ValueError("Bucketed sampling requires loaded audio.")
        sr = dataset.target_sr
        boundaries = [round(x * sr) / sr for x in self.boundaries]
        return BucketedBatchSampler(
            resolve_durations(dataset.rows, cache_path=self.duration_cache),
            batch_size,
            boundaries=boundaries,
            shuffle=shuffle,
            drop_last=drop_last,
            **self.options,
        )

    def build_collate_fn(self, dataset):
        return partial(
            collate_bucket,
            boundaries=tuple(round(x * dataset.target_sr) for x in self.boundaries),
            sample_rate=dataset.target_sr,
        )

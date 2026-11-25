"""Seeded source-subset selection through the native dataloader sampler."""

from typing import Protocol

import torch


def shuffled_sample_indices(size: int, max_samples: int | None, sample_seed: int = 0) -> list[int] | None:
    """Shuffle indices before limiting; None preserves ordinary dataset order.

    Uses a private RNG, without decoding samples or changing global RNG state.
    For the same dataset order and seed, larger limits extend the same shuffled
    prefix. Existing outputs still count toward the selected source subset.
    """
    if not isinstance(sample_seed, int) or isinstance(sample_seed, bool):
        raise ValueError("sample_seed must be an integer.")
    if max_samples is None:
        if sample_seed != 0:
            raise ValueError("sample_seed requires max_samples.")
        return None
    if not isinstance(max_samples, int) or isinstance(max_samples, bool) or max_samples < 1:
        raise ValueError("max_samples must be a positive integer or None.")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ValueError("Dataset size must be a nonnegative integer.")
    generator = torch.Generator().manual_seed(sample_seed)
    return torch.randperm(size, generator=generator)[:max_samples].tolist()


class LengthAwareBatchSampler(torch.utils.data.Sampler):
    """Pool-sort indices and shuffle batches using an epoch-local RNG."""

    def __init__(
        self,
        lengths,
        batch_size,
        *,
        max_batch_cost=None,
        max_batch_seconds=None,
        max_batch_attention_cost=None,
        pool_size=1024,
        cost_power=1,
        shuffle=True,
        drop_last=False,
        seed=0,
    ):
        import math

        self.lengths = tuple(float(x) for x in lengths)
        if any(not math.isfinite(x) or x <= 0 for x in self.lengths):
            raise ValueError("Lengths must be finite and positive.")
        if batch_size < 1 or pool_size < batch_size or cost_power not in (1, 2):
            raise ValueError("Require 1 <= batch_size <= pool_size and cost_power in (1, 2).")
        budgets = {1: max_batch_seconds, 2: max_batch_attention_cost}
        if max_batch_cost is not None:
            if budgets[cost_power] is not None:
                raise ValueError("Duplicate budget for the same cost power.")
            budgets[cost_power] = max_batch_cost
        self.budgets = {power: budget for power, budget in budgets.items() if budget is not None}
        for power, budget in self.budgets.items():
            if not math.isfinite(budget) or budget <= 0:
                raise ValueError("Batch cost must be finite and positive.")
            if any(x**power > budget for x in self.lengths):
                raise ValueError("An individual sample exceeds the batch cost budget.")
        if self.budgets and drop_last:
            raise ValueError("drop_last is ambiguous with dynamic batching.")
        self.batch_size = batch_size
        self.max_batch_cost = max_batch_cost
        self.pool_size = pool_size
        self.cost_power = cost_power
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    @property
    def sampler(self):
        """Expose the epoch-aware sampler through the BatchSampler protocol."""
        return self

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _batches(self):
        rng = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = (
            torch.randperm(len(self.lengths), generator=rng).tolist()
            if self.shuffle
            else list(range(len(self.lengths)))
        )
        ordered = []
        for start in range(0, len(indices), self.pool_size):
            ordered.extend(sorted(indices[start : start + self.pool_size], key=self.lengths.__getitem__))
        batches = []
        batch = []
        longest = 0.0
        for idx in ordered:
            length = self.lengths[idx]
            exceeds_budget = any(
                (len(batch) + 1) * max(longest, length) ** power > budget
                for power, budget in self.budgets.items()
            )
            if batch and (len(batch) == self.batch_size or exceeds_budget):
                batches.append(batch)
                batch = []
                longest = 0.0
            batch.append(idx)
            longest = max(longest, length)
        if batch and (not self.drop_last or len(batch) == self.batch_size):
            batches.append(batch)
        if self.shuffle:
            batches = [batches[i] for i in torch.randperm(len(batches), generator=rng).tolist()]
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        return len(self._batches())


def batching_statistics(lengths, batches):
    """Measure padding and attention cost without loading any samples."""
    valid = padded = attention = count = steps = 0
    for batch in batches:
        if not batch:
            continue
        values = [lengths[i] for i in batch]
        valid += sum(values)
        padded += len(values) * max(values)
        attention += len(values) * max(values) ** 2
        count += len(values)
        steps += 1
    return dict(
        examples=count,
        batches=steps,
        valid_volume=valid,
        padded_volume=padded,
        attention_cost=attention,
        padding_fraction=1 - valid / padded if padded else 0.0,
    )


class SamplingPolicy(Protocol):
    """Construct an index batch sampler from dataset metadata."""

    def build(self, dataset, *, batch_size: int, shuffle: bool, drop_last: bool) -> torch.utils.data.Sampler: ...


class LengthAwareSampling:
    """Configure duration-based batching independently of the dataset loader."""

    def __init__(
        self,
        pool_size: int = 1024,
        max_batch_seconds: float | None = None,
        max_batch_cost: float | None = None,
        cost_power: int = 1,
        duration_cache: str | None = None,
        seed: int = 0,
        max_batch_attention_cost: float | None = None,
    ):
        self.duration_cache = duration_cache
        self.options = dict(
            pool_size=pool_size,
            max_batch_cost=max_batch_cost,
            max_batch_seconds=max_batch_seconds,
            max_batch_attention_cost=max_batch_attention_cost,
            cost_power=cost_power,
            seed=seed,
        )

    def build(self, dataset, *, batch_size: int, shuffle: bool, drop_last: bool) -> LengthAwareBatchSampler:
        from .durations import resolve_durations

        if dataset.max_length is not None:
            raise ValueError("Length-aware sampling requires variable-length collation.")
        return LengthAwareBatchSampler(
            resolve_durations(dataset.rows, cache_path=self.duration_cache),
            batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            **self.options,
        )


class DistributedBatchSampler(torch.utils.data.Sampler):
    """Shard a deterministic global batch schedule without splitting batches."""

    def __init__(self, batch_sampler, *, rank: int, world_size: int, drop_remainder: bool = False):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid distributed rank or world size.")
        self.batch_sampler = batch_sampler
        self.rank = rank
        self.world_size = world_size
        self.drop_remainder = drop_remainder

    @property
    def sampler(self):
        return self

    def set_epoch(self, epoch):
        sampler = getattr(self.batch_sampler, "sampler", self.batch_sampler)
        advance = getattr(sampler, "set_epoch", None)
        if advance is not None:
            advance(epoch)

    def __len__(self):
        count = len(self.batch_sampler)
        if self.drop_remainder:
            return count // self.world_size
        return (count + self.world_size - 1) // self.world_size

    def __iter__(self):
        batches = list(self.batch_sampler)
        total = len(self) * self.world_size
        if batches and total > len(batches):
            batches = (batches * ((total + len(batches) - 1) // len(batches)))[:total]
        return iter(batches[self.rank : total : self.world_size])

"""Seeded source-subset selection through the native dataloader sampler."""

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

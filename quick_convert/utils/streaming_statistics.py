"""Mergeable streaming moments for corpus-level analyses."""

from __future__ import annotations

import math

import torch


class RunningMoments:
    """Population mean/variance without retaining observations."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, values):
        x = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
        if x.numel() == 0:
            return
        if not torch.isfinite(x).all():
            raise ValueError("Statistics require finite observations")
        n = x.numel()
        other = RunningMoments()
        other.count = n
        other.mean = x.mean().item()
        other.m2 = ((x - other.mean) ** 2).sum().item()
        self.merge(other)

    def merge(self, other: RunningMoments):
        if other.count == 0:
            return
        total = self.count + other.count
        delta = other.mean - self.mean
        self.m2 += other.m2 + delta * delta * self.count * other.count / total
        self.mean += delta * other.count / total
        self.count = total

    @property
    def variance(self):
        return self.m2 / self.count if self.count else math.nan

    @property
    def std(self):
        return math.sqrt(max(self.variance, 0)) if self.count else math.nan

    def summary(self):
        return {
            "count": self.count,
            "mean": self.mean if self.count else math.nan,
            "std": self.std,
            "variance": self.variance,
        }


class PrioritySample:
    """Exact extrema and uniform frame sampling via bounded random priorities."""

    def __init__(self, capacity, generator):
        self.capacity, self.generator = capacity, generator
        self.values = torch.empty(0, dtype=torch.float64)
        self.priorities = torch.empty(0, dtype=torch.float64)
        self.minimum, self.maximum = float("inf"), -float("inf")
        self.count = 0

    def update(self, values):
        if not values.numel():
            return
        self.minimum = min(self.minimum, values.min().item())
        self.maximum = max(self.maximum, values.max().item())
        self.count += values.numel()
        if not self.capacity:
            return
        for chunk in values.split(65536):
            priorities = torch.rand(len(chunk), generator=self.generator, dtype=torch.float64)
            if len(self.values) == self.capacity:
                candidates = priorities < self.priorities.max()
                chunk, priorities = chunk[candidates], priorities[candidates]
                if not len(chunk):
                    continue
            values = torch.cat((self.values, chunk))
            priorities = torch.cat((self.priorities, priorities))
            if len(values) > self.capacity:
                indices = priorities.topk(self.capacity, largest=False).indices
                values, priorities = values[indices], priorities[indices]
            self.values, self.priorities = values, priorities

    def edges(self, strategy, count):
        if strategy == "quantile":
            edges = torch.quantile(self.values, torch.arange(1, count, dtype=torch.float64) / count)
        else:
            edges = torch.linspace(self.minimum, self.maximum, count + 1, dtype=torch.float64)[1:-1]
        return torch.unique_consecutive(edges)


def average_ranks(values):
    """One-based ranks with average ties, computed only within one utterance."""
    _, inverse, counts = torch.unique(values, sorted=True, return_inverse=True, return_counts=True)
    ranks = counts.cumsum(0).double() - (counts.double() - 1) / 2
    return ranks[inverse]


class RunningCovariance:
    """Stable merge of paired moments, with optional within-utterance ranking."""

    def __init__(self):
        self.x, self.y = RunningMoments(), RunningMoments()
        self.cross = 0.0

    def update(self, x, y):
        if not len(x):
            return
        dx, dy = x - x.mean(), y - y.mean()
        old, n = self.x.count, len(x)
        self.cross += float((dx * dy).sum()) + (float(x.mean()) - self.x.mean) * (
            float(y.mean()) - self.y.mean
        ) * old * n / (old + n)
        self.x.update(x)
        self.y.update(y)

    def correlation(self):
        denominator = math.sqrt(max(self.x.m2, 0) * max(self.y.m2, 0))
        return max(-1.0, min(1.0, self.cross / denominator)) if denominator else None

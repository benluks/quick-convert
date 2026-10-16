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

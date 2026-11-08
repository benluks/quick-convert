"""Timestamped acoustic measurement contract."""

from dataclasses import dataclass
from typing import ClassVar

import torch


@dataclass(frozen=True)
class ProsodyMeasurements:
    """Named view of existing unpadded acoustic sidecars; physical units remain intact.

    Construction validates the on-disk schema once. Views preserve the caller's
    tensor; no peak normalization, copying, or loss conventions are applied.
    """

    values: torch.Tensor
    COLUMNS: ClassVar[tuple[str, ...]] = ("time_seconds", "f0_hz", "voiced", "rms", "voicing_score")

    def __post_init__(self):
        values = self.values
        if not isinstance(values, torch.Tensor) or not values.is_floating_point():
            raise TypeError("Prosody measurements must be a floating-point tensor.")
        if values.ndim != 2 or values.shape[1] != len(self.COLUMNS) or not len(values):
            raise ValueError("Expected nonempty [T, 5] prosody measurements.")
        if not torch.isfinite(values).all():
            raise ValueError("Prosody measurements must be finite.")
        if torch.any(self.timestamps < 0) or torch.any(self.timestamps[1:] <= self.timestamps[:-1]):
            raise ValueError("Prosody timestamps must be nonnegative and strictly increasing.")
        if torch.any((values[:, 2] != 0) & (values[:, 2] != 1)):
            raise ValueError("Voiced flags must be zero or one.")
        if torch.any(self.rms < 0) or torch.any(self.f0_hz < 0) or torch.any(self.f0_hz[self.voiced] <= 0):
            raise ValueError("RMS/F0 must be nonnegative and voiced F0 positive.")

    @property
    def timestamps(self):
        return self.values[:, 0]

    @property
    def f0_hz(self):
        return self.values[:, 1]

    @property
    def voiced(self):
        return self.values[:, 2].bool()

    @property
    def rms(self):
        return self.values[:, 3]

    @property
    def voicing_score(self):
        return self.values[:, 4]

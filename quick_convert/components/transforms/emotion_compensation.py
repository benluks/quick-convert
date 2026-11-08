from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn


class EmotionBoundaryCompensator(nn.Module):
    """Move an x-vector to a fixed signed distance from an emotion boundary.

    PSST1's post-hoc emotion compensation: the vector is rescaled to norm
    ``sqrt(192)`` and then shifted along the unit normal of the linear SVM
    boundary for its emotion until its signed distance equals
    :attr:`TARGET_DISTANCE` (negated for ``sad``). This is the closed form of
    the research code's ``linear_interpolate(start=30, end=40, steps=2)[1:]``.

    Args:
        boundaries_path: ``emotion_boundaries.pt`` mapping each label in
            :attr:`EMOTIONS` to a unit-norm ``[192]`` boundary normal.
    """

    EMOTIONS = ("neu", "hap", "sad", "ang")
    EMBEDDING_DIM = 192
    TARGET_DISTANCE = 40.0
    NEGATIVE_EMOTIONS = frozenset({"sad"})

    def __init__(self, boundaries_path: str | Path) -> None:
        super().__init__()
        boundaries = torch.load(boundaries_path, map_location="cpu", weights_only=True)
        missing = set(self.EMOTIONS) - set(boundaries)
        if missing:
            raise ValueError(f"{boundaries_path} is missing boundaries for {sorted(missing)}.")
        self.register_buffer(
            "normals",
            torch.stack([boundaries[label].float().reshape(self.EMBEDDING_DIM) for label in self.EMOTIONS]),
        )

    def forward(self, embedding: torch.Tensor, emotion: str) -> torch.Tensor:
        """Compensate one ``[192]`` embedding for ``emotion``."""
        if emotion not in self.EMOTIONS:
            raise ValueError(f"Unknown emotion {emotion!r}; expected one of {self.EMOTIONS}.")
        if embedding.shape != (self.EMBEDDING_DIM,):
            raise ValueError(f"Expected one embedding with shape [{self.EMBEDDING_DIM}], got {tuple(embedding.shape)}.")

        normal = self.normals[self.EMOTIONS.index(emotion)].to(embedding)
        distance = -self.TARGET_DISTANCE if emotion in self.NEGATIVE_EMOTIONS else self.TARGET_DISTANCE
        x = embedding / embedding.norm() * math.sqrt(self.EMBEDDING_DIM)
        return x + (distance - x @ normal) * normal

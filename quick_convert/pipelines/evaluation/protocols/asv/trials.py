from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ASVTrial:
    enroll_id: str
    test_id: str
    target: bool

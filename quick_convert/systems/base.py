"""Common inference loading inherited by task-level systems."""

from __future__ import annotations

from os import PathLike
from typing import Self

from torch import nn

from quick_convert.utils.device import DeviceLike


class InferenceSystem(nn.Module):
    """Task model with run and artifact restoration independent of training.

    Training compositions expose this model as top-level ``system`` and wrap
    it in their training module as ``self.system``. The inference layer owns
    restoration; subclasses need no checkpoint-loading implementation.
    """

    @classmethod
    def from_run(
        cls,
        run_dir: PathLike,
        *,
        checkpoint: PathLike = "checkpoints/last.ckpt",
        config: PathLike | None = None,
        map_location: DeviceLike = "cpu",
        strict: bool = True,
    ) -> Self:
        from quick_convert.inference import load_system_from_run

        system = load_system_from_run(
            run_dir, checkpoint=checkpoint, config=config, map_location=map_location, strict=strict
        )
        if not isinstance(system, cls):
            raise TypeError(f"Run restored {type(system).__name__}, expected {cls.__name__}.")
        return system

    @classmethod
    def from_artifact(cls, artifact_dir: PathLike, *, map_location: DeviceLike = "cpu", strict: bool = True) -> Self:
        from quick_convert.inference import load_inference_artifact

        system = load_inference_artifact(artifact_dir, map_location=map_location, strict=strict)
        if not isinstance(system, cls):
            raise TypeError(f"Artifact restored {type(system).__name__}, expected {cls.__name__}.")
        return system

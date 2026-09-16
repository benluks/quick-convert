"""Compatibility imports for the former Lightning base-module location."""

from quick_convert.training.lightning.modules.base import (
    BaseTrainingModule,
    StepOutputT,
    TrainingStepOutput,
)


__all__ = ["BaseTrainingModule", "StepOutputT", "TrainingStepOutput"]

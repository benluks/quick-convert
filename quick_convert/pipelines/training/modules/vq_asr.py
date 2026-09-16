"""Compatibility imports for the former VQ-ASR training-module location."""

from quick_convert.training.lightning.modules.vq_asr import (
    VQASROutput,
    VQASRTrainingModule,
)


__all__ = ["VQASROutput", "VQASRTrainingModule"]

"""Compatibility imports for the former SSL training-module location."""

from quick_convert.training.lightning.modules.ssl_reconstruction import (
    SSLReconstructionOutput,
    SSLReconstructionTrainingModule,
)


__all__ = ["SSLReconstructionOutput", "SSLReconstructionTrainingModule"]

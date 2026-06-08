from .asr_losses import CTCLoss
from .distil_losses import BaseDistilLoss, MSELoss, MaskedMSELoss
from .speaker_losses import BaseSpeakerLoss, AAMSoftmaxLoss, CosineDistanceLoss

__all__ = [
    "CTCLoss",
    "BaseDistilLoss",
    "MSELoss",
    "MaskedMSELoss",
    "BaseSpeakerLoss",
    "AAMSoftmaxLoss",
    "CosineDistanceLoss",
]
from .asr_losses import CTCLoss
from .distil_losses import BaseDistilLoss, MaskedMSELoss, MSELoss
from .speaker_losses import AAMSoftmaxLoss, BaseSpeakerLoss, CosineDistanceLoss


__all__ = [
    "CTCLoss",
    "BaseDistilLoss",
    "MSELoss",
    "MaskedMSELoss",
    "BaseSpeakerLoss",
    "AAMSoftmaxLoss",
    "CosineDistanceLoss",
]

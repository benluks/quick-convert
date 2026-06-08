import torch
import torch.nn as nn
import torch.nn.functional as F

from quick_convert.utils.masking import masked_loss


class BaseDistilLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, x: torch.FloatTensor, targets: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError("Subclasses must implement the forward method.")

class MSELoss(BaseDistilLoss):
    def __init__(self, reduction: str = 'mean'):
        super().__init__(reduction=reduction)

    def forward(self, x: torch.FloatTensor, targets: torch.FloatTensor) -> torch.FloatTensor:
        return F.mse_loss(x, targets, reduction=self.reduction)

class MaskedMSELoss(BaseDistilLoss):
    def __init__(self, reduction: str = 'mean'):
        super().__init__(reduction=reduction)

    def forward(self, x: torch.FloatTensor, targets: torch.FloatTensor, mask: torch.BoolTensor) -> torch.FloatTensor:
        return masked_loss(F.mse_loss, x, targets, mask, reduction=self.reduction)
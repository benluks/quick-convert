from ast import Tuple
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BaseSpeakerLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Unsupported reduction type: {reduction}")
        self.reduction = reduction

    def forward(
        self,
        speaker_features: torch.FloatTensor,
        speaker_embs: torch.FloatTensor = None,
        speaker_labels: torch.LongTensor = None,
    ) -> torch.Tensor:
        raise NotImplementedError("BaseSpeakerLoss is an abstract class and cannot be used directly.")


class CosineDistanceLoss(BaseSpeakerLoss):
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction)

    def forward(self, speaker_features: torch.FloatTensor, speaker_embs: torch.FloatTensor) -> torch.Tensor:
        loss = 1 - torch.cosine_similarity(speaker_features, speaker_embs, dim=1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class AAMSoftmaxLoss(BaseSpeakerLoss):
    """Additive angular margin softmax.

    Reference:
    ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    https://arxiv.org/pdf/1801.07698
    Adapted from ESPnet: https://github.com/espnet/espnet/blob/master/espnet2/spk/loss/aamsoftmax.py

    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        margin: float = 0.2,
        scale: float = 30.0,
        reduction: str = "mean",
    ):
        super().__init__(reduction)

        self.in_dim = in_dim
        self.num_classes = num_classes

        self.margin = margin
        self.scale = scale

        self.weight = torch.nn.Parameter(torch.FloatTensor(num_classes, in_dim), requires_grad=True)
        nn.init.xavier_normal_(self.weight, gain=1)

        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)  # Threshold for easy_margin
        self.mm = math.sin(math.pi - self.margin) * self.margin  # For numerical stability

        self.ce = nn.CrossEntropyLoss(reduction=self.reduction)

    def forward(
        self, speaker_features: torch.FloatTensor, speaker_labels: torch.LongTensor
    ) -> tuple[torch.Tensor, float, torch.Tensor]:
        # Apply linear transformation to get "cosine similarities"
        cosine = F.linear(F.normalize(speaker_features), F.normalize(self.weight))  # Output: (batch_size, nclasses)

        # Get predictions and accuracy for monitoring
        preds = torch.argmax(cosine, dim=1)
        speaker_labels = speaker_labels
        accuracy = (preds == speaker_labels).float().mean()

        # Convert cosine similarities to sine values using the identity sin^2 + cos^2 = 1
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))

        # Apply angular margin to positive samples:
        # cos(θ + m) = cosθ·cos(m) - sinθ·sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Standard margin with numerical stability
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Convert labels to one-hot encoding
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, speaker_labels.view(-1, 1), 1)

        output = (one_hot * phi) + (  # Ground truth: apply positive margin
            (1.0 - one_hot) * cosine
        )  # Other negatives: no margin

        # Scale logits for stable training
        output = output * self.scale

        loss = self.ce(output, speaker_labels)

        accuracy = accuracy.item()
        preds = preds.detach().cpu()
        return loss, accuracy, preds

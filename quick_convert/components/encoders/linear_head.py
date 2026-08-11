from dataclasses import replace

import torch
import torch.nn as nn

from quick_convert.components.layers.heads import HeadOutput, HeadTarget, SupervisedHead
from quick_convert.components.losses.distil_losses import BaseDistilLoss, MaskedMSELoss, MSELoss
from quick_convert.utils.masking import masked_loss


class LinearHead(SupervisedHead):
    """
    Simple linear head that applies a linear layer to the content encoder output.
    """

    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 128,
        loss: BaseDistilLoss = None,
    ):
        super().__init__()
        self.ln = nn.LayerNorm(input_dim)
        self.linear_head = nn.Linear(input_dim, output_dim)
        self.loss = loss or MaskedMSELoss("frame")

    def forward(self, content_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            content_features: (B, T, input_dim) output of the content encoder

        Returns:
            predicted_features: (B, T, output_dim)
        """
        return HeadOutput(predictions=self.linear_head(self.ln(content_features)))

    def predict(
        self,
        features: torch.Tensor,
        *,
        lengths: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> HeadOutput:
        del lengths, padding_mask
        return self.forward(features)

    def compute_loss(
        self,
        features,
        *,
        targets: HeadTarget,
        padding_mask,
        lengths=None,
    ) -> torch.Tensor:
        """Compute loss between predicted features and target features."""
        output = self.forward(features)

        if output.predictions.shape[1] != targets.values.shape[1]:
            raise ValueError(
                "Frame-level head prediction and target lengths differ: "
                f"{output.predictions.shape[1]} vs "
                f"{targets.values.shape[1]}."
            )

        return replace(output, loss=self.loss(output.predictions, targets.values, mask=padding_mask))

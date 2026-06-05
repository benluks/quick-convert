import torch
import torch.nn as nn

from quick_convert.utils.masking import masked_loss


class LinearHead(nn.Module):
    """
    Simple linear head that applies a linear layer to the content encoder output.
    """

    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 128,
    ):
        super().__init__()
        self.ln = nn.LayerNorm(input_dim)
        self.linear_head = nn.Linear(input_dim, output_dim)

    def forward(self, content_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            content_features: (B, T, output_dim) output of the content encoder

        Returns:
            prosody_features: (B, T, output_dim)
        """
        return self.linear_head(self.ln(content_features))

    def compute_loss(self, x: torch.FloatTensor, targets: torch.FloatTensor, mask: torch.LongTensor) -> torch.Tensor:
        """Compute MSE loss between predicted prosody features and target prosody features."""
        x = self.forward(x)

        return masked_loss(nn.functional.mse_loss, x, targets, mask)

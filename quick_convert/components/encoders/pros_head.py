import torch
import torch.nn as nn

class ProsodyHead(nn.Module):
    """
    Simple prosody head that applies a linear layer to the content encoder output.
    """

    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 128,
    ):
        super().__init__()
        self.prosody_head = nn.Linear(input_dim, output_dim)

    def forward(
            self, 
            content_features: torch.Tensor
        ) -> torch.Tensor:
        """
        Args:
            content_features: (B, T, output_dim) output of the content encoder

        Returns:
            prosody_features: (B, T, output_dim)
        """
        return self.prosody_head(content_features)
    
    def compute_loss(
            self, 
            prosody_features: torch.FloatTensor, 
            prosody_targets: torch.FloatTensor
        ) -> torch.Tensor:
        """ Compute MSE loss between predicted prosody features and target prosody features. """
        return nn.functional.mse_loss(prosody_features, prosody_targets)
import torch
import torch.nn.functional as F
from torch import nn


class LayerWeightedSum(nn.Module):
    """
    Weighted sum over multi-layer representation.
    Should you choose to use a projection at the end, you need to know the feature
    dimension beforehand
    """

    def __init__(self, num_layers: int, projection: nn.Linear | None = None) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(1, num_layers))
        self.projection = projection or nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape [B, T, L, D], got {tuple(x.shape)}.")
        if x.shape[2] != self.weights.shape[1]:
            raise ValueError(f"Expected {self.weights.shape[1]} representation layers, got {x.shape[2]}.")

        weights = F.softmax(self.weights, dim=-1)
        x = torch.einsum("btlc,kl->btc", x, weights)
        return self.projection(x)

    @staticmethod
    def output_lengths(input_lengths: torch.Tensor) -> torch.Tensor:
        """Layer fusion and its feature projection preserve the time axis."""
        return input_lengths

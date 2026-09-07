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
        weights = F.softmax(self.weights, dim=-1)
        x = torch.einsum("btlc,kl->btc", x, weights)
        return self.projection(x)

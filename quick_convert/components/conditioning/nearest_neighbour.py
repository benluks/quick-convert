"""Frame-level retrieval independent of speech encoders and decoders."""

import torch
from torch import nn
from torch.nn import functional as F


class CosineTopK(nn.Module):
    """Select reference frame indices by cosine similarity, in query chunks."""

    def __init__(self, k: int = 4, chunk_size: int = 1024):
        super().__init__()
        if k < 1 or chunk_size < 1:
            raise ValueError("k and chunk_size must be positive.")
        self.k = k
        self.chunk_size = chunk_size

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        if query.ndim != 2 or keys.ndim != 2 or query.shape[1] != keys.shape[1]:
            raise ValueError("Expected query and keys with matching frame dimensions.")
        if keys.shape[0] < self.k:
            raise ValueError("Reference pool contains fewer frames than k.")
        if not query.is_floating_point() or not keys.is_floating_point():
            raise ValueError("Retrieval requires floating-point features.")
        keys = F.normalize(keys.float(), dim=-1)
        if query.shape[0] == 0:
            return torch.empty((0, self.k), dtype=torch.long, device=query.device)
        return torch.cat(
            [
                (F.normalize(chunk.float(), dim=-1) @ keys.T).topk(self.k, dim=-1).indices
                for chunk in query.split(self.chunk_size)
            ]
        )


class MeanAggregation(nn.Module):
    """Average selected reference values without normalizing them."""

    def forward(self, values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return values[indices].mean(dim=-2)


class NearestNeighbourMatching(nn.Module):
    """Retrieve using keys and synthesize using aligned, optionally distinct values."""

    def __init__(self, selector: nn.Module | None = None, aggregation: nn.Module | None = None):
        super().__init__()
        self.selector = selector if selector is not None else CosineTopK()
        self.aggregation = aggregation if aggregation is not None else MeanAggregation()

    def forward(
        self, query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor | None = None
    ) -> torch.Tensor:
        values = keys if values is None else values
        if values.ndim != 2 or values.shape[0] != keys.shape[0]:
            raise ValueError("Reference keys and values must have aligned frames.")
        return self.aggregation(values, self.selector(query, keys))

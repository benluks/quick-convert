
import torch
import torch.nn as nn
from typing import Tuple

# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Computes cos/sin tables on-the-fly for any sequence length and applies
    the rotation to query and key tensors.

    Args:
        dim:  Head dimension (must be even).
        base: Base frequency (default 10000).
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: (B, H, T, D)
            k: (B, H, T, D)
        Returns:
            Rotated q and k, same shapes.
        """
        cos, sin = self._cos_sin(q.size(2), q.device, q.dtype)
        return self._rotate(q, cos, sin), self._rotate(k, cos, sin)

    def _cos_sin(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cache is not None and self._cache[0] == seq_len:
            _, cached_dtype, cached_device, cos, sin = self._cache
            # Recompute only needed if seq_len changed; otherwise cast/move in place.
            if cached_device != device:
                cos, sin = cos.to(device=device), sin.to(device=device)
            if cached_dtype != dtype:
                cos, sin = cos.to(dtype=dtype), sin.to(dtype=dtype)
            if cached_device != device or cached_dtype != dtype:
                self._cache = (seq_len, dtype, device, cos, sin)
            return cos, sin

        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)               # (T, D/2)
        emb = torch.cat([freqs, freqs], dim=-1)             # (T, D)
        cos = emb.cos()[None, None].to(dtype)
        sin = emb.sin()[None, None].to(dtype)
        self._cache = (seq_len, dtype, device, cos, sin)
        return cos, sin

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1) * sin + x * cos

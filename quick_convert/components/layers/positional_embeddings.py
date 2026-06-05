
import math

import torch
import torch.nn as nn
from typing import Optional, Tuple

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
        """Build (or retrieve from cache) the cos/sin rotation tables.

        Results are cached by ``(seq_len, dtype, device)``; a cache hit casts
        or moves tensors in-place rather than recomputing.

        Args:
            seq_len: Number of positions to generate tables for.
            device:  Target device.
            dtype:   Target floating-point dtype.

        Returns:
            cos: shape ``(1, 1, seq_len, dim)``
            sin: shape ``(1, 1, seq_len, dim)``
        """
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
        """Apply the RoPE rotation to a single tensor.

        Args:
            x:   Input tensor, shape ``(B, H, T, D)``.
            cos: Cosine table, shape ``(1, 1, T, D)``.
            sin: Sine table,   shape ``(1, 1, T, D)``.

        Returns:
            Rotated tensor, same shape as ``x``.
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1) * sin + x * cos



# ---------------------------------------------------------------------------
# Sinusoidal Positional Embedding for timesteps from Matcha TTS
# ---------------------------------------------------------------------------
# Taken from Chatterbox's MatchaTTS: 
# https://github.com/resemble-ai/chatterbox/blob/master/src/chatterbox/models/s3gen/matcha/decoder.py 
# Used to embed time step in diffusion/flow matching models
# Embeds time step t to a vector of size dim using 
# sin/cos functions of different frequencies.
class SinusoidalPosEmb(torch.nn.Module):
    """
    Sinusoidal embedding for diffusion timesteps.

    Maps a batch of scalar timestep values to fixed-size embedding vectors
    using log-spaced sin/cos frequencies. Intended for conditioning diffusion
    models on the current noise level ``t``.

    Args:
        dim: Embedding dimension (must be even).
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x: torch.Tensor, scale: float = 1000) -> torch.Tensor:
        """
        Args:
            x:     Scalar timestep values, shape ``(B,)`` or ``()``.
            scale: Multiplier applied before computing frequencies. Since
                   diffusion timesteps are typically in ``[0, 1]``, scaling
                   by 1000 spreads them into a range where the sin/cos values
                   vary meaningfully across steps.

        Returns:
            Embedding tensor of shape ``(B, dim)``, where the first ``dim//2``
            columns are sin components and the last ``dim//2`` are cos.
        """
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample
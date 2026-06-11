from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

# Heavily inspired by "https://github.com/BUTSpeechFIT/DiariZen/blob/main/diarizen/models/module/conformer.py"

from .positional_embeddings import RoPE


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head self-attention layer with optional padding mask.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        dropout: float = 0.0,
        pos_emb_base: float = 10000.0,
        use_sdpa: bool = True,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("`embed_dim` must be divisible by `num_heads`.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.use_sdpa = use_sdpa
        self.pos_emb = RoPE(self.head_dim, base=pos_emb_base)

        # Per-head Q/K/V projections: (num_heads, embed_dim, head_dim).
        # Stored as Parameters so all heads are projected in a single batched matmul.
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, D = x.shape

        # 1. Linear projections and reshape for multi-head attention
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)

        # 2. Apply positional embeddings
        q, k = self.pos_emb(q, k)

        # 3. Scaled dot-product attention (flash-capable path via PyTorch SDPA)
        attn_mask = padding_mask.unsqueeze(1).unsqueeze(2) if padding_mask is not None else None

        if self.use_sdpa and hasattr(F, "scaled_dot_product_attention"):
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)  # (B, H, T_q, T_k)
            if attn_mask is not None:
                attn_scores = attn_scores.masked_fill(~attn_mask, float("-inf"))
            attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, H, T, T)
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, v)  # (B, H, T, D)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)

        # 4. Final linear projection
        output = self.out_proj(attn_output)  # (B, T, D)
        return output

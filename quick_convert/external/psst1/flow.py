# Adapted from the PSST1 research code (flow_matching/flow_matcher.py), which
# adapts https://github.com/facebookresearch/DiT. Local changes: emotion
# conditioning branches removed (PSST1 uses the unconditional flow) and the
# Euler integrators rewritten as functions.

import math

import torch
import torch.nn.functional as F
from torch import nn


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class DiTBlock(nn.Module):
    """A DiT block with adaLN-Zero conditioning; MLPs replace attention."""

    def __init__(self, hidden_size, cond_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4), nn.SiLU(), nn.Linear(hidden_size * 4, hidden_size)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4), nn.SiLU(), nn.Linear(hidden_size * 4, hidden_size)
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * hidden_size, bias=True))
        self.linear_final = nn.Linear(hidden_size, hidden_size)
        self.linear_inter = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, c):
        shift_1, scale_1, gate_1, shift_2, scale_2, gate_2 = self.adaLN_modulation(c).chunk(6, dim=1)
        x_1 = x + gate_1 * self.mlp1(modulate(self.norm1(x), shift_1, scale_1))
        return (
            self.linear_final(x)
            + self.linear_inter(x_1)
            + gate_2 * self.mlp2(modulate(self.norm2(x_1), shift_2, scale_2))
        )


class VelocityNet(nn.Module):
    """U-Net MLP for speaker embeddings: 192 -> 96 -> 48 -> 24 -> 48 -> 96 -> 192."""

    def __init__(self, input_dim=192, cond_dim=256):
        super().__init__()
        self.time_gen = TimestepEmbedder(cond_dim)
        self.enc1 = nn.Linear(input_dim, 96)
        self.block1 = DiTBlock(96, cond_dim)
        self.enc2 = nn.Linear(96, 48)
        self.block2 = DiTBlock(48, cond_dim)
        self.enc3 = nn.Linear(48, 24)
        self.bottleneck = DiTBlock(24, cond_dim)
        self.dec1 = nn.Linear(24, 48)
        self.block_up1 = DiTBlock(48, cond_dim)
        self.dec2 = nn.Linear(48, 96)
        self.block_up2 = DiTBlock(96, cond_dim)
        self.out_proj = nn.Linear(96, input_dim)

    def forward(self, x, t):
        t_emb = self.time_gen(t)
        h1 = self.block1(F.silu(self.enc1(x)), t_emb)
        h2 = self.block2(F.silu(self.enc2(h1)), t_emb)
        h3 = self.bottleneck(F.silu(self.enc3(h2)), t_emb)
        x_up = self.block_up1(F.silu(self.dec1(h3)), t_emb) + h2
        x_up = self.block_up2(F.silu(self.dec2(x_up)), t_emb) + h1
        return self.out_proj(x_up)


def integrate_forward(model, x0, steps):
    """Euler integration from noise (t=0) to embedding (t=1)."""
    xt = x0
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((xt.shape[0],), i / steps, device=xt.device)
        xt = xt + model(xt, t) * dt
    return xt


def integrate_backward(model, x1, steps):
    """Euler integration from embedding (t=1) back to noise (t=0)."""
    xt = x1
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((xt.shape[0],), 1.0 - i * dt, device=xt.device)
        xt = xt - model(xt, t) * dt
    return xt

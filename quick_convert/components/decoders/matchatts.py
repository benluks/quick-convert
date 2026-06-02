import einops

import torch
import torch.nn as nn

from typing import Optional, Tuple

from layers import (
    SinusoidalPosEmb,
    TimestepEmbedding,
    TransformerBlock,
    Conv1DBlock,
    ResnetBlock1D,
    Downsample1D,
    Upsample1D
)

def _apply_attn(x: torch.Tensor, mask_b1t: torch.Tensor, attn: TransformerBlock) -> torch.Tensor:
    """Apply a TransformerBlock; inputs/outputs in (B, C, T) layout."""
    x = einops.rearrange(x, "b c t -> b t c")
    # `mask_b1t` is a keep-mask (1=valid, 0=pad). MHA expects padding mask (True=pad).
    padding_mask = einops.rearrange(mask_b1t, "b 1 t -> b t") == 0
    x = attn(x, attention_mask=padding_mask)
    return einops.rearrange(x, "b t c -> b c t")

class DownBlock1D(nn.Module):
    """ResNet + Attention + downsample (or passthrough conv when is_last=True)."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        time_emb_dim: int,
        num_heads: int,
        dropout: float,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.resnet    = ResnetBlock1D(dim=dim_in, dim_out=dim_out, time_emb_dim=time_emb_dim)
        self.attn      = TransformerBlock(dim=dim_out, num_attention_heads=num_heads, dropout=dropout)

        if self.is_last:
            self.downsample = nn.Conv1d(dim_out, dim_out, 3, padding=1) 
        else:
            self.downsample = Downsample1D(dim_out)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (next_x, skip, next_mask) at matching temporal resolution."""
        x    = self.resnet(x, mask, t)
        x    = _apply_attn(x, mask, self.attn)
        skip = x
        x    = self.downsample(x * mask)
        next_mask = mask if self.is_last else mask[:, :, ::2]
        return x, skip, next_mask


class MidBlock1D(nn.Module):
    """ResNet + Attention at the bottleneck (no change in resolution)."""

    def __init__(self, dim: int, time_emb_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.resnet = ResnetBlock1D(dim=dim, dim_out=dim, time_emb_dim=time_emb_dim)
        self.attn   = TransformerBlock(dim=dim, num_attention_heads=num_heads, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x, mask, t)
        x = _apply_attn(x, mask, self.attn)
        return x


class UpBlock1D(nn.Module):
    """Skip-concat + ResNet + Attention + upsample (or passthrough conv when is_last=True)."""

    def __init__(
        self,
        dim: int,
        time_emb_dim: int,
        num_heads: int,
        dropout: float,
        is_last: bool = False,
    ):
        super().__init__()
        self.resnet   = ResnetBlock1D(dim=2 * dim, dim_out=dim, time_emb_dim=time_emb_dim)
        self.attn     = TransformerBlock(dim=dim, num_attention_heads=num_heads, dropout=dropout)
        self.upsample = nn.Conv1d(dim, dim, 3, padding=1) if is_last else Upsample1D(dim, use_conv_transpose=True)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, mask: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        x = einops.pack([x[:, :, :skip.shape[-1]], skip], "b * t")[0]
        x = self.resnet(x, mask, t)
        x = _apply_attn(x, mask, self.attn)
        x = self.upsample(x * mask)
        return x


class Decoder(nn.Module):
    # Fixed architecture:
    #   channels:        [256, 256]
    #   n_blocks:        1   (transformer block per resnet)
    #   num_mid_blocks:  2
    #   act_fn:          snakebeta

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: int = 256,
        dropout: float = 0.05,
        num_heads: int = 2,
        act_fn: str = "snakebeta",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        ch = channels
        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = ch * 4
        self.time_mlp = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=time_embed_dim,
            act_fn="silu",
        )

        block_kwargs = dict(time_emb_dim=time_embed_dim, num_heads=num_heads, dropout=dropout)

        self.down_0 = DownBlock1D(dim_in=in_channels, dim_out=ch, **block_kwargs)
        self.down_1 = DownBlock1D(dim_in=ch, dim_out=ch, is_last=True, **block_kwargs)

        self.mid_0  = MidBlock1D(dim=ch, **block_kwargs)
        self.mid_1  = MidBlock1D(dim=ch, **block_kwargs)

        self.up_0   = UpBlock1D(dim=ch, **block_kwargs)
        self.up_1   = UpBlock1D(dim=ch, is_last=True, **block_kwargs)

        self.final_block = Conv1DBlock(ch, ch)
        self.final_proj  = nn.Conv1d(ch, self.out_channels, 1)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        *kwargs
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, in_channels, T)
            mask: (B, 1, T)
            mu:   (B, in_channels, T)  mean / conditioning signal
            t:    (B,)                 diffusion timestep
            spks: (B, spk_channels)    optional speaker embedding
            cond: unused placeholder
        Returns:
            (B, out_channels, T)
        """
        t = self.time_mlp(self.time_embeddings(t))

        x = einops.pack([x, mu], "b * t")[0]
        if spks is not None:
            spks = einops.repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = einops.pack([x, spks], "b * t")[0]

        # ---- Down --------------------------------------------------------
        x, skip0, mask1 = self.down_0(x, mask, t)
        x, skip1, _     = self.down_1(x, mask1, t)

        # ---- Mid ---------------------------------------------------------
        x = self.mid_0(x, mask1, t)
        x = self.mid_1(x, mask1, t)

        # ---- Up ----------------------------------------------------------
        x = self.up_0(x, skip1, mask1, t)
        x = self.up_1(x, skip0, mask, t)

        # ---- Output ------------------------------------------------------
        x = self.final_block(x, mask)

        return self.final_proj(x * mask) * mask
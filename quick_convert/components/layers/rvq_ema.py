from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from quick_convert.components.layers.rvq import BaseResidualVectorQuantizer, RVQLosses, RVQOutput

from .vector_quantize import VectorQuantize


class ResidualVectorQuantizerEMA(BaseResidualVectorQuantizer):
    """
    Residual vector quantizer based on vector-quantize-pytorch.

    Unlike ResidualVectorQuantizerDAC, this implementation uses an
    EMA-updated codebook with optional k-means initialization and dead-code
    replacement.

    Input/output interface intentionally matches ResidualVectorQuantizerDAC.

    Input:
        z:
            [B, D, T]

        padding_mask:
            [B, T, 1] or [B, T], where True denotes a valid timestep.

    Output:
        RVQOutput
            z_q:
                [B, D, T]

            layer_z_qs:
                list[[B, D, T]]

            codes:
                [B, Q, T]

            latents:
                [B, Q * codebook_dim, T]
                Residual input presented to each quantizer.

            loss.commitment_loss:
                Scalar sum of per-layer commitment losses.

            loss.codebook_loss:
                Zero for this implementation because codebooks are updated
                through EMA rather than an explicit gradient loss.
    """

    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 8,
        codebook_size: int = 1024,
        codebook_dim: int | None = None,
        *,
        decay: float = 0.8,
        kmeans_init: bool = True,
        kmeans_iters: int = 10,
        threshold_ema_dead_code: int = 2,
        use_cosine_sim: bool = True,
        sync_codebook: bool = False,
        quantizer_dropout: float = 0.0,
        quant_grad_frac: float = 0.0,
        loss_weights: dict[str, float] | None = None,
        **vq_kwargs: Any,
    ):
        super().__init__()

        if codebook_dim is None:
            codebook_dim = input_dim

        if not 0.0 <= quantizer_dropout <= 1.0:
            raise ValueError(f"quantizer_dropout must be in [0, 1], got {quantizer_dropout}.")

        if not 0.0 <= quant_grad_frac <= 1.0:
            raise ValueError(f"quant_grad_frac must be in [0, 1], got {quant_grad_frac}.")

        self.input_dim = input_dim
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.quantizer_dropout = quantizer_dropout
        self.quant_grad_frac = quant_grad_frac

        self.project_in = nn.Linear(input_dim, codebook_dim) if input_dim != codebook_dim else nn.Identity()

        self.project_out = nn.Linear(codebook_dim, input_dim) if input_dim != codebook_dim else nn.Identity()
        self.loss_weights = loss_weights or {
            "commitment": 1.0,
            "orthogonal_reg": 0.0,
            "codebook_diversity": 0.0,
        }

        self.enabled_losses = {name for name, weight in self.loss_weights.items() if weight != 0.0}

        self.quantizers = nn.ModuleList(
            [
                VectorQuantize(
                    dim=codebook_dim,
                    codebook_size=codebook_size,
                    codebook_dim=codebook_dim,
                    decay=decay,
                    kmeans_init=kmeans_init,
                    kmeans_iters=kmeans_iters,
                    threshold_ema_dead_code=threshold_ema_dead_code,
                    use_cosine_sim=use_cosine_sim,
                    # ----------
                    # VectorQuantize uses these both as enable flags and weights.
                    # We only use them to enable calculation; actual weighting
                    # belongs to ResidualVectorQuantizerEMA.
                    commitment_weight=float(self.loss_weights.get("commitment", 0.0) != 0.0),
                    orthogonal_reg_weight=float(self.loss_weights.get("orthogonal_reg", 0.0) != 0.0),
                    codebook_diversity_loss_weight=float(self.loss_weights.get("codebook_diversity", 0.0) != 0.0),
                    # ----------
                    ema_update=True,
                    learnable_codebook=False,
                    sync_codebook=sync_codebook,
                    **vq_kwargs,
                )
                for _ in range(n_codebooks)
            ]
        )
        if any(quantizer.in_place_codebook_optimizer is not None for quantizer in self.quantizers):
            self.enabled_losses.add("inplace_optimize")
        self.log_inplace_optimize = any(
            quantizer.in_place_codebook_optimizer is not None for quantizer in self.quantizers
        )

    @staticmethod
    def _frac_gradient(
        x: torch.Tensor,
        fraction: float,
    ) -> torch.Tensor:
        """
        Control how strongly later residual quantizers backpropagate through
        earlier quantized outputs.

        fraction = 0:
            residual subtraction uses detached quantized values, matching
            vector-quantize-pytorch ResidualVQ's default.

        fraction = 1:
            full gradient through residual subtraction.
        """
        if fraction <= 0.0:
            return x.detach()

        if fraction >= 1.0:
            return x

        return fraction * x + (1.0 - fraction) * x.detach()

    @property
    def codebooks(self) -> torch.Tensor:
        """
        Return codebooks as [Q, K, D].
        """
        return torch.stack(
            [quantizer.codebook for quantizer in self.quantizers],
            dim=0,
        )

    def _project_layer_out(
        self,
        quantized: torch.Tensor,
        *,
        include_bias: bool,
    ) -> torch.Tensor:
        """
        Project a single quantizer contribution from codebook_dim back into
        input_dim.

        Bias is included exactly once across the whole RVQ stack so that

            sum(layer_z_qs) == z_q

        remains true.
        """
        if isinstance(self.project_out, nn.Identity):
            return quantized

        bias = self.project_out.bias if include_bias else None

        return F.linear(
            quantized,
            self.project_out.weight,
            bias=bias,
        )

    def forward(
        self,
        z: torch.Tensor,
        padding_mask: torch.Tensor,
        n_quantizers: int | None = None,
    ) -> RVQOutput:
        """
        Args:
            z:
                [B, D, T]

            padding_mask:
                [B, T, 1] or [B, T], True for valid positions.

            n_quantizers:
                Optional number of active quantizers during evaluation.

        Returns:
            RVQOutput compatible with ResidualVectorQuantizerDAC.
        """
        if z.ndim != 3:
            raise ValueError(f"Expected z with shape [B, D, T], got {z.shape}.")

        if padding_mask.ndim == 3:
            padding_mask = padding_mask.squeeze(-1)

        if padding_mask.ndim != 2:
            raise ValueError(f"Expected padding_mask with shape [B, T] or [B, T, 1], got {padding_mask.shape}.")

        valid_mask = padding_mask.bool()

        # Current code uses [B, D, T].
        # vector-quantize-pytorch uses [..., D].
        x = z.transpose(1, 2)  # [B, T, D]

        x = self.project_in(x)  # [B, T, codebook_dim]

        residual = x
        quantized_sum = torch.zeros_like(x)

        layer_quantized: list[torch.Tensor] = []
        codebook_indices: list[torch.Tensor] = []
        residual_latents: list[torch.Tensor] = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks

        if not 1 <= n_quantizers <= self.n_codebooks:
            raise ValueError(f"n_quantizers must be in [1, {self.n_codebooks}], got {n_quantizers}.")

        # Match the online ResidualVQ behavior: when quantizer dropout is
        # enabled, choose one cutoff for this batch.
        active_quantizers = n_quantizers

        if self.training and self.quantizer_dropout > 0.0:
            if torch.rand((), device=z.device) < self.quantizer_dropout:
                active_quantizers = int(
                    torch.randint(
                        1,
                        self.n_codebooks + 1,
                        (),
                        device=z.device,
                    ).item()
                )
            else:
                active_quantizers = self.n_codebooks

        quantizer_loss_breakdowns: dict[str, list[torch.Tensor]] = {}

        for layer_idx, quantizer in enumerate(self.quantizers):
            if layer_idx >= active_quantizers:
                break

            # This is analogous to the per-layer projected latents returned
            # by the DAC implementation.
            residual_latents.append(residual)

            quantized, indices, _, loss_breakdown = quantizer(
                residual,
                mask=valid_mask,
                freeze_codebook=not self.training,
                return_loss_breakdown=True,
            )
            for name, value in loss_breakdown._asdict().items():
                enabled = name in self.enabled_losses or (name == "inplace_optimize" and self.log_inplace_optimize)

                if not enabled:
                    continue
                quantizer_loss_breakdowns.setdefault(name, []).append(value)

            layer_quantized.append(quantized)
            codebook_indices.append(indices)

            quantized_sum = quantized_sum + quantized

            residual = residual - self._frac_gradient(
                quantized,
                self.quant_grad_frac,
            )

        # Project the summed representation once.
        z_q = self.project_out(quantized_sum)  # [B, T, input_dim]

        # The router expects individual additive RVQ contributions in the
        # original model dimension.
        #
        # project_out has a bias, so that bias must appear in exactly one
        # layer rather than once per quantizer.
        layer_z_qs = []

        for layer_idx, quantized in enumerate(layer_quantized):
            layer_output = self._project_layer_out(
                quantized,
                include_bias=(layer_idx == 0),
            )

            # Existing router expects [B, D, T].
            layer_z_qs.append(layer_output.transpose(1, 2))

        codes = torch.stack(
            codebook_indices,
            dim=1,
        )  # [B, Q, T]

        latents = torch.cat(
            [latent.transpose(1, 2) for latent in residual_latents],
            dim=1,
        )  # [B, Q * codebook_dim, T]

        raw_losses = {name: torch.stack(values).sum() for name, values in quantizer_loss_breakdowns.items()}
        weighted_losses = {
            name: self.loss_weights[name] * value for name, value in raw_losses.items() if name in self.loss_weights
        }
        loss = sum(
            weighted_losses.values(),
            start=z.new_zeros(()),
        )
        return RVQOutput(
            z_q=z_q.transpose(1, 2),
            layer_z_qs=layer_z_qs,
            codes=codes,
            latents=latents,
            loss=RVQLosses(
                loss=loss,
                raw=raw_losses,
                weighted=weighted_losses,
            ),
        )

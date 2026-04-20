from abc import ABC

import torch
import torch.nn.functional as F


def expand_time_like(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Expand t of shape (B,) so it broadcasts over x.
    """
    return t.view(t.shape[0], *([1] * (x.ndim - 1)))


class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        solver,
        sigma_min=1e-4,
        estimator=None,
    ):
        super().__init__()
        self.solver = solver
        self.sigma_min = sigma_min
        self.estimator = estimator

    @torch.inference_mode()
    def forward(self, x_init, n_timesteps, *, mask=None, **conditions):
        """
        Sampling / ODE solve.

        Args:
            x_init (torch.Tensor):
                Conditioning-shaped tensor used only to define output shape/device.
                In Matcha this was `mu`.
            n_timesteps (int):
                Number of Euler steps.
            mask (torch.Tensor | None):
                Optional broadcastable mask.
            **conditions:
                Arbitrary conditioning passed to estimator.

        Returns:
            torch.Tensor:
                Sampled tensor with same shape as x_init.
        """
        z = torch.randn_like(x_init)
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=x_init.device, dtype=x_init.dtype)
        return self.solve_euler(z, t_span=t_span, mask=mask, **conditions)

    def solve_euler(self, x, t_span, *, mask=None, **conditions):
        t = t_span[0]

        for step in range(1, len(t_span)):
            dt = t_span[step] - t
            dphi_dt = self.estimator(x, t.expand(x.shape[0]), mask=mask, **conditions)
            x = x + dt * dphi_dt
            t = t_span[step]

        return x

    def compute_loss(self, x1, *, x0=None, mask=None, **conditions):
        """
        Compute conditional flow matching loss.

        Args:
            x1 (torch.Tensor):
                Target tensor of shape (B, ...).
            x0 (torch.Tensor | None):
                Optional source noise. If None, sampled as Gaussian.
            mask (torch.Tensor | None):
                Optional broadcastable mask for the loss / estimator.
            **conditions:
                Arbitrary conditioning passed to estimator.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                loss, x_t
        """
        b = x1.shape[0]

        # random timestep
        t = torch.rand((b,), device=x1.device, dtype=x1.dtype)
        t_b = expand_time_like(t, x1)

        # sample noise p(x_0)
        if x0 is None:
            x0 = torch.randn_like(x1)

        # Matcha-style affine path
        x_t = (1 - (1 - self.sigma_min) * t_b) * x0 + t_b * x1
        u = x1 - (1 - self.sigma_min) * x0

        pred = self.estimator(x_t, t, mask=mask, **conditions)

        loss = F.mse_loss(pred, u, reduction="none")

        if mask is not None:
            mask_exp = mask.expand_as(loss)
            loss = (loss * mask_exp).sum() / mask_exp.sum().clamp_min(1)
        else:
            loss = loss.mean()

        return loss, x_t

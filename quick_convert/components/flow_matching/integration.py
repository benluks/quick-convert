"""Fixed-step Euler integration for PSST1 velocity fields.

Time construction and update order preserve the checkpoint recipe in both directions.
"""

import torch


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

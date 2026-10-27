from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from quick_convert.external.psst1.flow import VelocityNet, integrate_backward, integrate_forward

from .base import BaseSpeakerGenerator


def mix_latents(z_orig: torch.Tensor, z_rand: torch.Tensor, w: float) -> torch.Tensor:
    """Blend an inverted speaker latent with noise while keeping unit variance.

    ``w`` is the weight of the original speaker. When both inputs follow
    ``N(0, I)`` the result does too, so the flow maps it back to a plausible
    speaker.
    """
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"w must be in [0, 1], got {w}.")
    return ((1.0 - w) * z_rand + w * z_orig) / math.sqrt((1.0 - w) ** 2 + w**2)


class PSST1FlowSpeakerGenerator(BaseSpeakerGenerator):
    """Unconditional flow-matching x-vector generator from PSST1.

    Embeddings are standardized per dimension with the training statistics
    stored in the checkpoint, mapped to the flow's Gaussian prior by reverse
    Euler integration, blended with Gaussian noise (:func:`mix_latents`), and
    integrated forward again.

    Two departures from the research code are deliberate: noise is Gaussian
    (the research code drew uniform noise although the flow was trained from
    Gaussian noise), and standardization is per dimension (the research code
    applied the first dimension's mean and standard deviation to every
    dimension).

    Training follows the PSST1 system description
    (https://www.voiceprivacychallenge.org/vp2026/docs/T02_system_description.pdf): fit the standardization on the training
    x-vectors with :meth:`fit_standardization`, then minimize
    :meth:`compute_loss`. :meth:`save` writes the ``flow.pt`` format that the
    constructor loads.

    Args:
        checkpoint_path: ``flow.pt`` holding ``velocity_net``, ``mean`` and
            ``std``. ``None`` builds an untrained network with identity
            standardization, for training.
    """

    EMBEDDING_DIM = 192
    COND_DIM = 256

    def __init__(self, checkpoint_path: str | Path | None = None) -> None:
        super().__init__(embedding_dim=self.EMBEDDING_DIM)
        self.velocity_net = VelocityNet(input_dim=self.EMBEDDING_DIM, cond_dim=self.COND_DIM)
        self.register_buffer("mean", torch.zeros(self.EMBEDDING_DIM))
        self.register_buffer("std", torch.ones(self.EMBEDDING_DIM))
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.velocity_net.load_state_dict(checkpoint["velocity_net"])
            self.mean.copy_(checkpoint["mean"].float().reshape(self.EMBEDDING_DIM))
            self.std.copy_(checkpoint["std"].float().reshape(self.EMBEDDING_DIM))
            self.eval()

    @torch.no_grad()
    def fit_standardization(self, embeddings: torch.Tensor) -> None:
        """Set the per-dimension mean and population std from ``[N, 192]`` training x-vectors."""
        embeddings = embeddings.reshape(-1, self.EMBEDDING_DIM).double()
        if embeddings.shape[0] < 2:
            raise ValueError("Standardization needs at least two embeddings.")
        std = embeddings.std(dim=0, correction=0)
        self.mean.copy_(embeddings.mean(dim=0).to(self.mean))
        self.std.copy_(torch.where(std == 0, torch.ones_like(std), std).to(self.std))

    def save(self, path: str | Path) -> None:
        """Write the ``flow.pt`` inference checkpoint."""
        torch.save(
            {
                "velocity_net": {key: value.detach().cpu() for key, value in self.velocity_net.state_dict().items()},
                "mean": self.mean.detach().cpu(),
                "std": self.std.detach().cpu(),
            },
            path,
        )

    def standardize(self, embeddings: torch.Tensor) -> torch.Tensor:
        return (embeddings - self.mean) / self.std

    def destandardize(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings * self.std + self.mean

    @torch.inference_mode()
    def invert(self, embeddings: torch.Tensor, steps: int) -> torch.Tensor:
        """Map standardized embeddings ``[B, 192]`` to prior latents."""
        return integrate_backward(self.velocity_net, embeddings, steps)

    @torch.inference_mode()
    def generate(self, latents: torch.Tensor, steps: int) -> torch.Tensor:
        """Map prior latents ``[B, 192]`` to standardized embeddings."""
        return integrate_forward(self.velocity_net, latents, steps)

    @torch.inference_mode()
    def anonymize(
        self,
        embeddings: torch.Tensor,
        *,
        w: float,
        steps: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return pseudo-speaker embeddings for ``[192]`` or ``[B, 192]`` inputs.

        Noise is drawn on the CPU from ``generator`` so a seed gives the same
        result on every device.
        """
        single = embeddings.ndim == 1
        x = embeddings.reshape(-1, self.EMBEDDING_DIM).to(self.mean)
        z_orig = self.invert(self.standardize(x), steps)
        z_rand = torch.randn(z_orig.shape, generator=generator).to(z_orig)
        pseudo = self.destandardize(self.generate(mix_latents(z_orig, z_rand, w), steps))
        return pseudo[0] if single else pseudo

    def compute_loss(self, embeddings: torch.Tensor, **conditions) -> dict[str, torch.Tensor]:
        if conditions:
            raise ValueError(f"PSST1 flow is unconditional; got conditions {sorted(conditions)}.")
        x1 = self.standardize(embeddings)
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=x1.device)
        xt = (1.0 - t[:, None]) * x0 + t[:, None] * x1
        return {"loss": F.mse_loss(self.velocity_net(xt, t), x1 - x0)}

    @torch.inference_mode()
    def sample(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
        *,
        steps: int = 10,
        **conditions,
    ) -> torch.Tensor:
        if conditions:
            raise ValueError(f"PSST1 flow is unconditional; got conditions {sorted(conditions)}.")
        z = torch.randn(batch_size, self.EMBEDDING_DIM, device=device, dtype=dtype or self.mean.dtype)
        return self.destandardize(self.generate(z, steps))

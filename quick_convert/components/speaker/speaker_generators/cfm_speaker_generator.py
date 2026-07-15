from __future__ import annotations

import torch

from .base import BaseSpeakerGenerator


class CFMSpeakerGenerator(BaseSpeakerGenerator):
    def __init__(self, cfm: torch.nn.Module, embedding_dim: int) -> None:
        super().__init__(embedding_dim=embedding_dim)
        self.cfm = cfm

    def compute_loss(
        self,
        embeddings: torch.Tensor,
        **conditions,
    ) -> dict[str, torch.Tensor]:
        loss, x_t = self.cfm.compute_loss(embeddings, **conditions)
        return {
            "loss": loss,
            "x_t": x_t,
        }

    @torch.inference_mode()
    def sample(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
        **conditions,
    ) -> torch.Tensor:
        conditions = dict(conditions)
        dtype = dtype or torch.float32
        n_timesteps = conditions.pop("n_timesteps", 32)

        template = torch.zeros(
            batch_size,
            self.embedding_dim,
            device=device,
            dtype=dtype,
        )

        return self.cfm(template, n_timesteps=n_timesteps, **conditions)

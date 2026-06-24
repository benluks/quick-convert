from __future__ import annotations

import abc
from typing import Any, Optional

import torch
import lightning as L

from quick_convert.components.encoders import RVQDisentangler
from quick_convert.data.types import AudioBatch


class BaseEncoderDecoderTrainingModule(L.LightningModule):
    """
    Model-agnostic PyTorch Lightning base trainer for end-to-end
    encoder-decoder type anonymization models.

    Subclasses must implement :meth:`_shared_step`, which contains the
    model-specific forward pass and loss computation.

    Args:
        encoder:
            The encoder component. Stored as ``self.encoder``.
        decoder:
            Optional decoder component. Stored as ``self.decoder``.
        lr:
            Peak learning rate for AdamW.
        weight_decay:
            AdamW weight decay.
        cosine_t_max:
            ``T_max`` (number of steps per cosine half-cycle) for
            :class:`~torch.optim.lr_scheduler.CosineAnnealingLR`.
            Set to 0 to disable the scheduler.
    """

    def __init__(
        self,
        encoder: RVQDisentangler,
        decoder: torch.nn.Module,
        optimizer: type[torch.optim.Optimizer] = torch.optim.AdamW,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder", "tokenizer", "content_encoder"])

        self.encoder = encoder
        self.decoder = decoder
        self.optimizer_factory = optimizer
        self.lr_scheduler_factory = lr_scheduler

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _shared_step(self, batch: AudioBatch, stage: str) -> torch.Tensor:
        """Compute losses for *batch* and log them under *stage*/.

        Must return the scalar total loss tensor.
        """

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def training_step(self, batch: AudioBatch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: AudioBatch, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

        # ------------------------------------------------------------------
        # Optimizers / schedulers
        # ------------------------------------------------------------------

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]

        optimizer = self.optimizer_factory(params=params)

        if self.lr_scheduler_factory is None:
            return optimizer

        scheduler = self.lr_scheduler_factory(optimizer=optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

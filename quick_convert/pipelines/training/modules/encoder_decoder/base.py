from __future__ import annotations

import abc
from typing import Any, Optional

import torch
import lightning as L

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
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        lr_scheduler_cls: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        lr_scheduler_kwargs: Optional[dict[str, Any]] = None,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.AdamW,
        optimizer_kwargs: dict[str, Any] = {"weight_decay": 1e-2},
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder"])

        self.encoder = encoder
        self.decoder = decoder
        self.configure_optimizers = self._configure_optimizers(optimizer_cls, 
                                                               optimizer_kwargs, 
                                                               lr_scheduler_cls,
                                                               lr_scheduler_kwargs)
        

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

    def configure_optimizers(self) -> dict[str, Any]:
        params = [p for p in self.parameters() if p.requires_grad]

        optimizer = self.hparams.optimizer_cls(
            params,
            lr=self.hparams.lr,
            **self.hparams.optimizer_kwargs,
        )

        scheduler = self.hparams.lr_scheduler_cls(
            optimizer, 
            **self.hparams.lr_scheduler_kwargs
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

# quick_convert/pipelines/training/modules/base.py

from __future__ import annotations

import abc
from typing import Protocol, TypeVar

import lightning as L
import torch

from quick_convert.data import AudioBatch, BaseDataset
from quick_convert.utils import configure_device

from ..optim.base import Optimization


class TrainingStepOutput(Protocol):
    loss: torch.Tensor


StepOutputT = TypeVar(
    "StepOutputT",
    bound=TrainingStepOutput,
)


StepOutputT = TypeVar("StepOutputT", bound=TrainingStepOutput)


class BaseTrainingModule(L.LightningModule, abc.ABC):
    """Base class for models trained with quick-convert.

    Subclasses implement :meth:`_shared_step`, which is shared between
    training and validation and returns an object containing at least a
    scalar ``loss`` tensor.

    Optimizer and scheduler construction is delegated to
    :class:`Optimization`, allowing training modules to remain independent
    of the surrounding experiment configuration.

    Dataset-dependent initialization may be implemented in
    :meth:`setup_training`. This method is called by
    :class:`LightningTrainer` before the Lightning trainer is constructed.

    Args:
        optimization:
            Optimizer, scheduler, and optional warmup configuration.
    """

    def __init__(
        self,
        optimization: Optimization,
    ) -> None:
        super().__init__()

        self.optimization = optimization

    def setup_training(self, train_dataset: BaseDataset) -> None:
        """Perform dataset-dependent initialization before training.

        Subclasses may override this to fit indexers, construct output layers,
        build dataset-dependent losses, or perform other initialization that
        requires access to the training dataset.
        """
        return

    @abc.abstractmethod
    def _shared_step(
        self,
        batch: AudioBatch,
        stage: str,
    ) -> TrainingStepOutput:
        """Run a model-specific training or validation step."""

    def training_step(
        self,
        batch: AudioBatch,
        batch_idx: int,
    ) -> torch.Tensor:
        output = self._shared_step(batch, "train")
        return output.loss

    def validation_step(
        self,
        batch: AudioBatch,
        batch_idx: int,
    ) -> torch.Tensor:
        output = self._shared_step(batch, "val")
        self.log_validation_output(
            batch=batch,
            output=output,
            batch_idx=batch_idx,
        )
        return output.loss

    def on_validation_epoch_end(self):
        return super().on_validation_epoch_end()

    def log_validation_output(
        self,
        batch: AudioBatch,
        output: TrainingStepOutput,
        batch_idx: int,
    ) -> None:
        """Optionally log qualitative validation outputs."""

    def configure_optimizers(self):
        return self.optimization.configure(
            parameters=self.parameters(),
            total_steps=self.trainer.estimated_stepping_batches,
        )

    def load_for_inference(
        self,
        checkpoint_path,
        map_location: str | torch.device | None = None,
        strict: bool = True,
    ) -> tuple[list[str], list[str]]:
        """Load a Lightning checkpoint and prepare the module for inference."""

        checkpoint = torch.load(
            checkpoint_path,
            weights_only=False,
            map_location=configure_device(map_location),
        )

        state_dict = {key.replace("._orig_mod.", "."): value for key, value in checkpoint["state_dict"].items()}

        missing, unexpected = self.load_state_dict(
            state_dict,
            strict=strict,
        )

        self.eval()
        self.freeze()

        return missing, unexpected

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Remove names introduced by ``torch.compile`` from saved parameters."""

        checkpoint["state_dict"] = {
            key.replace("._orig_mod.", "."): value for key, value in checkpoint["state_dict"].items()
        }

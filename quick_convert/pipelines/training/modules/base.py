# quick_convert/pipelines/training/modules/base.py

from __future__ import annotations

import abc
from collections.abc import Iterable, Mapping
from os import PathLike
from pathlib import Path
from typing import Protocol, Self, TypeVar

import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn
from torch.optim import Optimizer

from quick_convert.data import AudioBatch, BaseDataset
from quick_convert.utils.device import DeviceLike, configure_device, override_devices

from ..logging.media_logger import MediaLogger, make_media_logger
from ..optim.base import Optimization


class TrainingStepOutput(Protocol):
    loss: torch.Tensor


StepOutputT = TypeVar(
    "StepOutputT",
    bound=TrainingStepOutput,
)


StepOutputT = TypeVar("StepOutputT", bound=TrainingStepOutput)


class BaseTrainingModule(L.LightningModule, abc.ABC):
    """Base Lightning module for trainable systems.

    Subclasses implement :meth:`_shared_step` and return an object containing
    at least a scalar ``loss`` tensor. The concrete output may contain any
    additional model-specific values needed for logging, validation, inference,
    or qualitative inspection.

    Args:
        optimizer:
            Callable that constructs an optimizer when passed ``params``.
            This is typically provided through Hydra using ``_partial_: true``.
        lr_scheduler:
            Optional callable that constructs a scheduler when passed
            ``optimizer``. This is also typically a Hydra partial.
    """

    def __init__(
        self,
        optimization: Optimization,
        checkpoint_exclude_prefixes: Iterable[str] = (),
        enable_grad_norm_logging: bool = True,
    ) -> None:
        super().__init__()

        self.optimization = optimization
        self.checkpoint_exclude_prefixes = tuple(
            self._normalize_checkpoint_prefix(prefix) for prefix in checkpoint_exclude_prefixes
        )
        self.enable_grad_norm_logging = enable_grad_norm_logging

    @classmethod
    def from_run(
        cls,
        run_dir: PathLike,
        *,
        checkpoint: PathLike = "checkpoints/last.ckpt",
        config: PathLike = "config.yaml",
        map_location: DeviceLike = None,
        strict: bool = True,
    ) -> Self:
        run_dir = Path(run_dir)

        cfg = OmegaConf.load(run_dir / config)
        module_cfg = cfg.pipeline.trainer.module

        map_location = configure_device(map_location)
        override_devices(module_cfg, str(map_location))

        model = instantiate(module_cfg)

        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = run_dir / checkpoint_path

        state = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )

        model.load_state_dict(
            state["state_dict"],
            strict=strict,
        )

        return model.to(map_location)

    @staticmethod
    def _normalize_checkpoint_prefix(prefix: str) -> str:
        """
        Normalize a module path for prefix matching.

        Both ``online_encoders`` and ``online_encoders.`` become
        ``online_encoders.``.
        """
        return prefix.rstrip(".") + "."

    @property
    def media_logger(self) -> MediaLogger:
        if not hasattr(self, "_media_logger"):
            self._media_logger = make_media_logger(self.logger)

        return self._media_logger

    def _is_checkpoint_excluded(self, key: str) -> bool:
        return key.startswith(self.checkpoint_exclude_prefixes)

    def _clean_compiled_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {key.replace("._orig_mod.", "."): value for key, value in state_dict.items()}

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        state_dict = self._clean_compiled_state_dict(checkpoint["state_dict"])

        checkpoint["state_dict"] = {
            key: value for key, value in state_dict.items() if not self._is_checkpoint_excluded(key)
        }

        # Record this for transparency/debugging. Loading does not need to rely
        # on it because the current module configuration is authoritative.
        checkpoint["checkpoint_exclude_prefixes"] = list(self.checkpoint_exclude_prefixes)

    def _prepare_checkpoint_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Prepare a checkpoint for strict loading.

        Excluded components are expected to have been reconstructed by the
        module configuration. Their current state is inserted so that strict
        loading does not report those keys as missing.
        """
        state_dict = self._clean_compiled_state_dict(state_dict)

        if not self.checkpoint_exclude_prefixes:
            return state_dict

        current_state = self._clean_compiled_state_dict(self.state_dict())

        for key, value in current_state.items():
            if self._is_checkpoint_excluded(key):
                state_dict.setdefault(key, value)

        return state_dict

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["state_dict"] = self._prepare_checkpoint_state_dict(checkpoint["state_dict"])

    def load_for_inference(
        self,
        checkpoint_path,
        map_location: str | torch.device = "cpu",
        strict: bool = True,
    ) -> tuple[list[str], list[str]]:
        checkpoint = torch.load(
            checkpoint_path,
            weights_only=False,
            map_location=map_location,
        )

        state_dict = self._prepare_checkpoint_state_dict(checkpoint["state_dict"])

        missing, unexpected = self.load_state_dict(
            state_dict,
            strict=strict,
        )

        self.eval()
        self.freeze()

        return missing, unexpected

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

    def log_grad_norms(
        self,
        modules: Mapping[str, nn.Module] | None = None,
        *,
        norm_type: float = 2.0,
        prefix: str = "grad_norm",
    ) -> None:
        modules = modules or self.grad_norm_modules

        metrics: dict[str, torch.Tensor] = {}

        for name, module in modules.items():
            parameters = [
                parameter for parameter in module.parameters() if parameter.requires_grad and parameter.grad is not None
            ]

            if not parameters:
                continue

            metrics[f"{prefix}/{name}"] = torch.nn.utils.get_total_norm(
                [parameter.grad for parameter in parameters],
                norm_type=norm_type,
            )

        if metrics:
            self.log_dict(
                metrics,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=False,
            )

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        if self.enable_grad_norm_logging:
            self.log_grad_norms()

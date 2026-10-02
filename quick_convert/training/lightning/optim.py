# quick_convert/training/lightning/optim.py

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    LinearLR,
    LRScheduler,
    SequentialLR,
)


Parameter = torch.nn.Parameter


class ClampedCosineAnnealingLR(torch.optim.lr_scheduler.CosineAnnealingLR):
    """Cosine decay that stays at eta_min after T_max scheduler steps."""

    def get_lr(self) -> list[float]:
        if self.last_epoch >= self.T_max:
            return [self.eta_min for _ in self.base_lrs]
        return super().get_lr()

    def _get_closed_form_lr(self) -> list[float]:
        if self.last_epoch >= self.T_max:
            return [self.eta_min for _ in self.base_lrs]
        return super()._get_closed_form_lr()


class OptimizerFactory(Protocol):
    def __call__(
        self,
        params: Iterable[Parameter],
        **kwargs: Any,
    ) -> Optimizer: ...


class SchedulerFactory(Protocol):
    def __call__(
        self,
        optimizer: Optimizer,
        **kwargs: Any,
    ) -> LRScheduler: ...


class WarmupProtocol(Protocol):
    """Construct or prepend a warmup phase."""

    @property
    def steps(self) -> int | float:
        """Absolute warmup steps or a fraction of total training steps."""

    def apply(
        self,
        optimizer: Optimizer,
        scheduler: LRScheduler | None = None,
    ) -> LRScheduler:
        """Return the complete scheduler, including warmup."""
        ...


@dataclass
class LinearWarmup:
    """Linear learning-rate warmup.

    ``steps`` may be either:

    - An integer giving the absolute number of optimizer steps.
    - A float in ``(0, 1]`` giving a fraction of total optimizer steps.

    The learning rate begins at ``optimizer_lr * start_factor`` and reaches
    the optimizer's configured learning rate at the end of warmup.

    If another scheduler is supplied, it begins after warmup.
    """

    steps: int | float
    start_factor: float = 1e-6

    def __post_init__(self) -> None:
        if isinstance(self.steps, bool):
            raise TypeError("Warmup steps must be an int or float, not bool.")

        if isinstance(self.steps, int):
            if self.steps <= 0:
                raise ValueError(f"Warmup steps must be positive, got {self.steps}.")

        elif isinstance(self.steps, float):
            if not 0.0 < self.steps <= 1.0:
                raise ValueError(
                    f"A floating-point warmup duration must be a fraction in the interval (0, 1], got {self.steps}."
                )

        else:
            raise TypeError(f"Warmup steps must be an int or float, got {type(self.steps).__name__}.")

        if not 0.0 < self.start_factor <= 1.0:
            raise ValueError(f"start_factor must be in the interval (0, 1], got {self.start_factor}.")

    def resolve_steps(
        self,
        total_steps: int | None,
    ) -> int:
        """Resolve the configured duration into an absolute step count."""

        if isinstance(self.steps, int):
            return self.steps

        if total_steps is None:
            raise ValueError("total_steps is required when warmup steps are specified as a fraction.")

        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}.")

        return max(1, round(total_steps * self.steps))

    def apply(
        self,
        optimizer: Optimizer,
        scheduler: LRScheduler | None = None,
    ) -> LRScheduler:
        total_steps = getattr(optimizer, "total_steps", None)
        warmup_steps = self.resolve_steps(total_steps)

        warmup = LinearLR(
            optimizer,
            start_factor=self.start_factor,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        if scheduler is None:
            return warmup

        return SequentialLR(
            optimizer,
            schedulers=[warmup, scheduler],
            milestones=[warmup_steps],
        )


@dataclass
class Optimization:
    """Build an optimizer and its complete learning-rate schedule."""

    optimizer: OptimizerFactory = torch.optim.AdamW
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)

    lr_scheduler: SchedulerFactory | None = torch.optim.lr_scheduler.CosineAnnealingLR
    lr_scheduler_kwargs: dict[str, Any] = field(default_factory=dict)

    warmup: WarmupProtocol | None = None

    interval: str = "step"
    frequency: int = 1
    monitor: str | None = None

    def configure(
        self,
        parameters: Iterable[Parameter],
        total_steps: int | None = None,
    ) -> Optimizer | dict[str, Any]:
        trainable_parameters = [parameter for parameter in parameters if parameter.requires_grad]

        optimizer = self.optimizer(
            trainable_parameters,
            **self.optimizer_kwargs,
        )

        if total_steps is not None:
            if (
                isinstance(total_steps, bool)
                or not math.isfinite(total_steps)
                or total_steps <= 0
                or int(total_steps) != total_steps
            ):
                raise ValueError(f"total_steps must be positive, got {total_steps}.")

            optimizer.total_steps = total_steps

        scheduler = self._build_scheduler(optimizer)

        if scheduler is None:
            return optimizer

        scheduler_config: dict[str, Any] = {
            "scheduler": scheduler,
            "interval": self.interval,
            "frequency": self.frequency,
        }

        if self.monitor is not None:
            scheduler_config["monitor"] = self.monitor

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }

    def _build_scheduler(
        self,
        optimizer: Optimizer,
    ) -> LRScheduler | None:
        scheduler: LRScheduler | None = None

        if self.lr_scheduler is not None:
            scheduler_factory = self.lr_scheduler
            scheduler_kwargs = dict(self.lr_scheduler_kwargs)
            if scheduler_kwargs.get("T_max") == "auto":
                if scheduler_factory is not torch.optim.lr_scheduler.CosineAnnealingLR:
                    raise ValueError("T_max='auto' requires torch.optim.lr_scheduler.CosineAnnealingLR.")
                if self.interval != "step" or self.frequency != 1:
                    raise ValueError("T_max='auto' requires interval='step' and frequency=1.")
                total_steps = getattr(optimizer, "total_steps", None)
                if total_steps is None:
                    raise ValueError("T_max='auto' requires a finite total_steps training budget.")
                warmup_steps = 0
                if self.warmup is not None:
                    if not isinstance(self.warmup, LinearWarmup):
                        raise ValueError("T_max='auto' supports LinearWarmup or no warmup.")
                    warmup_steps = self.warmup.resolve_steps(total_steps)
                decay_steps = int(total_steps) - warmup_steps
                if decay_steps <= 0:
                    raise ValueError("Warmup must leave at least one step for cosine decay.")
                scheduler_kwargs["T_max"] = decay_steps
                scheduler_factory = ClampedCosineAnnealingLR
            scheduler = scheduler_factory(
                optimizer,
                **scheduler_kwargs,
            )

        if self.warmup is not None:
            scheduler = self.warmup.apply(
                optimizer=optimizer,
                scheduler=scheduler,
            )

        return scheduler

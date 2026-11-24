"""Ordered, named optimization phases for Lightning training."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch

from ..optim import Optimization


class PhasedOptimizationMixin:
    """One optimizer update per named phase, in configuration order.

    Subclasses supply parameter groups and phase losses. Each loss is computed
    after preceding updates, with only the active optimizer's parameters enabled.
    """

    def _init_phases(self, optimization: Mapping[str, Optimization]):
        if not optimization:
            raise ValueError("At least one optimization phase is required.")
        self.automatic_optimization = False
        self.phase_names = tuple(optimization)
        self._phase_schedulers = {}

    def phase_parameters(self) -> Mapping[str, Iterable[torch.nn.Parameter]]:
        raise NotImplementedError

    def phase_step(self, batch, phase: str):
        raise NotImplementedError

    def configure_phase_optimizers(self):
        groups = self.phase_parameters()
        if set(groups) != set(self.phase_names):
            raise ValueError("Parameter groups must match optimization phase names.")
        seen = set()
        optimizers, schedulers = [], []
        self._phase_schedulers = {}
        for name in self.phase_names:
            parameters = [p for p in groups[name] if p.requires_grad]
            ids = {id(p) for p in parameters}
            if not ids or len(ids) != len(parameters) or seen & ids:
                raise ValueError(f"Empty, duplicate or overlapping parameter group: {name}")
            seen.update(ids)
            config = self.optimization[name]
            if config.monitor is not None or config.frequency != 1:
                raise ValueError("Phased optimization currently requires unmonitored schedulers with frequency=1.")
            if config.interval not in ("step", "epoch"):
                raise ValueError("Scheduler interval must be step or epoch.")
            if config.warmup is not None or any(v == "auto" for v in config.lr_scheduler_kwargs.values()):
                raise ValueError("Automatic step budgets and warmup are not yet supported for phased optimization.")
            result = config.configure(parameters)
            if isinstance(result, dict):
                optimizers.append(result["optimizer"])
                self._phase_schedulers[name] = len(schedulers)
                schedulers.append(result["lr_scheduler"])
            else:
                optimizers.append(result)
        if seen != {id(p) for p in self.parameters() if p.requires_grad}:
            raise ValueError("Every trainable parameter must belong to exactly one phase.")
        return optimizers, schedulers

    def validate_phase_settings(self):
        if self.trainer.accumulate_grad_batches != 1:
            raise ValueError("Phased optimization does not support gradient accumulation yet.")
        if self.trainer.gradient_clip_val:
            raise ValueError("Phased optimization does not support trainer gradient clipping yet.")

    def _step_scheduler(self, name, interval):
        if name not in self._phase_schedulers or self.optimization[name].interval != interval:
            return
        schedulers = self.lr_schedulers()
        if not isinstance(schedulers, (list, tuple)):
            schedulers = [schedulers]
        schedulers[self._phase_schedulers[name]].step()

    def execute_phases(self, batch):
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        losses = {}
        for name, optimizer in zip(self.phase_names, optimizers, strict=True):
            self.toggle_optimizer(optimizer)
            try:
                optimizer.zero_grad()
                output = self.phase_step(batch, name)
                self.manual_backward(output.loss)
                optimizer.step()
                optimizer.zero_grad()
                losses[name] = output.loss.detach()
            finally:
                self.untoggle_optimizer(optimizer)
            self._step_scheduler(name, "step")
        return losses

    def step_epoch_phase_schedulers(self):
        for name in self.phase_names:
            self._step_scheduler(name, "epoch")

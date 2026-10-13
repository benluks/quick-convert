"""Resolve logger factories before initialization and persist their run identities."""

import re
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger


_RUNS_KEY = "quick_convert_logger_runs"


def _wandb_identity(checkpoint_path, options):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    runs = [run for run in checkpoint.get(_RUNS_KEY, []) if run.get("backend") == "wandb"]
    if len(runs) > 1:
        runs = [
            run
            for run in runs
            if all(options.get(key) is None or options[key] == run.get(key) for key in ("entity", "project"))
        ]
        if len(runs) != 1:
            raise ValueError("Checkpoint has ambiguous or unmatched W&B identities; set an explicit logger id.")
    if len(runs) == 1:
        identity = {key: runs[0].get(key) for key in ("id", "entity", "project")}
        if not identity["id"] or not identity["project"]:
            raise ValueError("Checkpoint contains an incomplete W&B run identity.")
        for key in ("entity", "project"):
            if options.get(key) is not None and identity[key] is not None and options[key] != identity[key]:
                raise ValueError(
                    f"Checkpoint W&B {key} differs from the logger; set an explicit logger id to override."
                )
        return {key: value for key, value in identity.items() if value is not None}
    path = Path(checkpoint_path).absolute()
    # Legacy Quick Convert run directories are <project>/<8-character run ID>/checkpoints/.
    if path.parent.name == "checkpoints" and re.fullmatch(r"[a-zA-Z0-9]{8}", path.parent.parent.name):
        project = path.parent.parent.parent.name
        if options.get("project") == project:
            return {"project": project, "id": path.parent.parent.name}
    raise ValueError("Cannot recover the W&B run identity; set logger id and project explicitly, or resume='never'.")


def resolve_loggers(loggers, checkpoint_path=None):
    """Construct deferred loggers, resuming W&B identity when a checkpoint is supplied.

    Explicit IDs and resume modes take precedence. Already constructed W&B
    loggers must have an explicit ID for resume; use a partial for automatic recovery.
    Other logger types and disabled/default logging retain Lightning's behavior.
    """
    if loggers is None or isinstance(loggers, bool):
        return loggers
    multiple = isinstance(loggers, Sequence)
    resolved = []
    for logger in loggers if multiple else [loggers]:
        if isinstance(logger, partial):
            if checkpoint_path and isinstance(logger.func, type) and issubclass(logger.func, WandbLogger):
                if logger.args:
                    raise ValueError("Use keyword arguments for automatic W&B checkpoint resume.")
                options = dict(logger.keywords)
                if options.get("resume", "must") in ("must", "allow"):
                    if not (options.get("id") or options.get("version")):
                        identity = _wandb_identity(checkpoint_path, options)
                        for key, value in identity.items():
                            if options.get(key) is None:
                                options[key] = value
                    options.setdefault("resume", "must")
                logger = partial(logger.func, **options)
            logger = logger()
        elif checkpoint_path and isinstance(logger, WandbLogger) and not logger.version:
            raise ValueError("Automatic W&B resume requires a deferred logger: partial(WandbLogger, ...).")
        resolved.append(logger)
    return resolved if multiple else resolved[0]


class LoggerRunCheckpoint(Callback):
    """Keep external logger identity in training checkpoints, outside model state."""

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        runs = []
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                run = logger.experiment
                runs.append({"backend": "wandb", "id": run.id, "entity": run.entity, "project": run.project})
        if runs:
            checkpoint[_RUNS_KEY] = runs

# quick_convert/systems/training/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseTrainer(ABC):
    """
    Base interface for objects that know how to train something.

    A trainer backend owns training execution, while the pipeline owns I/O and
    workflow orchestration.

    Examples:
        - LightningTrainer
        - TokenizerTrainer
        - SpeechBrainTrainer
    """

    def __init__(
        self,
        # don't set output dir for now. It causes problems because it has to refer circularly to pipeline,
        # but pipeline has to instantiate trainer.
        output_dir: str | Path | None = None,
    ) -> None:
        self.output_dir = Path(output_dir) if output_dir is not None else None

    @abstractmethod
    def train(
        self,
        train_dataset: Any | None = None,
        val_dataset: Any | None = None,
        # test_dataset: Any | None = None,
        # out_dir: str | Path | None = None,
        # **kwargs: Any,
    ) -> Any:
        """
        Run training.

        The pipeline should call this method after preparing/choosing datasets
        and resolving the output directory.

        Args:
            train_dataset:
                Dataset used for training, if applicable.
            val_dataset:
                Optional validation dataset.
            test_dataset:
                Optional test dataset.
            output_dir:
                Optional runtime output directory. If provided, this should
                usually override `self.output_dir`.
            **kwargs:
                Escape hatch for backend-specific runtime options.

        Returns:
            Backend-specific training result.
        """
        raise NotImplementedError

    def resolve_output_dir(
        self,
        out_dir: str | Path | None = None,
    ) -> Path | None:
        """
        Resolve the effective output directory.

        Runtime `output_dir` takes precedence over the constructor value.
        """
        if out_dir is not None:
            return Path(out_dir)

        return self.output_dir

"""Lightning adapter for training speaker-embedding generators."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import lightning as L
import torch

from quick_convert.components.speaker.speaker_generators.base import BaseSpeakerGenerator
from quick_convert.data import BaseDataset
from quick_convert.data.types import AudioBatch

from ..optim import Optimization
from .base import BaseTrainingModule


@dataclass
class SpeakerGeneratorOutput:
    loss: torch.Tensor


class SpeakerGeneratorTrainingModule(BaseTrainingModule):
    """Train a :class:`BaseSpeakerGenerator` on precomputed speaker embeddings.

    Each batch reads the ``embedding_resource`` tensor resource and minimizes
    the generator's ``compute_loss``. Before training, a generator that defines
    ``fit_standardization`` receives every training embedding so it can fit
    its normalization statistics on the training set.

    Args:
        optimization: Optimizer and learning-rate schedule.
        generator: The speaker generator to train.
        embedding_resource: Name of the per-sample embedding resource.
        seed: Seed for Lightning's global RNGs, set before fitting and
            training; ``None`` leaves them untouched.
        export_path: Where to call ``generator.save`` when training ends;
            ``None`` skips the export.
        stats_batch_size: Batch size used to read embeddings for fitting.
    """

    def __init__(
        self,
        optimization: Optimization,
        generator: BaseSpeakerGenerator,
        embedding_resource: str = "speaker_embedding",
        seed: int | None = None,
        export_path: PathLike | None = None,
        stats_batch_size: int = 1024,
        **kwargs: Any,
    ) -> None:
        super().__init__(optimization=optimization, **kwargs)
        self.generator = generator
        self.embedding_resource = embedding_resource
        self.seed = seed
        self.export_path = Path(export_path) if export_path is not None else None
        self.stats_batch_size = stats_batch_size

    @property
    def grad_norm_modules(self):
        return {"generator": self.generator}

    def _embeddings(self, batch: AudioBatch) -> torch.Tensor:
        if self.embedding_resource not in batch.resources:
            raise KeyError(f"Batch has no {self.embedding_resource!r} resource; add it to the dataset load list.")
        values = batch.resources[self.embedding_resource].values
        return values.reshape(len(batch), self.generator.embedding_dim).float()

    def setup_training(self, train_dataset: BaseDataset) -> None:
        if self.seed is not None:
            L.seed_everything(self.seed, workers=True)
        if not hasattr(self.generator, "fit_standardization"):
            return

        loader = train_dataset.make_dataloader(batch_size=self.stats_batch_size, shuffle=False)
        embeddings = torch.cat([self._embeddings(batch) for batch in loader])
        self.generator.fit_standardization(embeddings)

    def _shared_step(self, batch: AudioBatch, stage: str) -> SpeakerGeneratorOutput:
        losses = self.generator.compute_loss(self._embeddings(batch))
        loss = losses["loss"]
        self.log(
            f"{stage}/loss",
            loss,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=True,
            batch_size=len(batch),
        )
        return SpeakerGeneratorOutput(loss=loss)

    def on_train_end(self) -> None:
        if self.export_path is None or not self.trainer.is_global_zero:
            return
        self.export_path.parent.mkdir(parents=True, exist_ok=True)
        self.generator.save(self.export_path)

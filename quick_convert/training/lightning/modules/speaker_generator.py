"""Lightning adapter for training speaker-embedding generators."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path

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
        system: The speaker generator to train.
        embedding_resource: Name of the per-sample embedding resource.
        export_path: Where to call ``generator.save`` when training ends;
            ``None`` skips the export.
        stats_batch_size: Batch size used to read embeddings for fitting.
    """

    def __init__(
        self,
        optimization: Optimization,
        system: BaseSpeakerGenerator,
        embedding_resource: str = "speaker_embedding",
        export_path: PathLike | None = None,
        stats_batch_size: int = 1024,
        enable_grad_norm_logging: bool = True,
    ) -> None:
        super().__init__(optimization=optimization, enable_grad_norm_logging=enable_grad_norm_logging)
        self.system = system
        self.embedding_resource = embedding_resource
        self.export_path = Path(export_path) if export_path is not None else None
        self.stats_batch_size = stats_batch_size

    @property
    def grad_norm_modules(self):
        return {"generator": self.system}

    def _embeddings(self, batch: AudioBatch) -> torch.Tensor:
        if self.embedding_resource not in batch.resources:
            raise KeyError(f"Batch has no {self.embedding_resource!r} resource; add it to the dataset load list.")
        values = batch.resources[self.embedding_resource].values
        if values.shape != (len(batch), 1, self.system.embedding_dim):
            raise ValueError("Speaker generator training requires one embedding vector per sample.")
        return values[:, 0].float()

    def setup_training(self, train_dataset: BaseDataset) -> None:
        if not hasattr(self.system, "fit_standardization"):
            return

        loader = train_dataset.make_dataloader(batch_size=self.stats_batch_size, shuffle=False)
        embeddings = torch.cat([self._embeddings(batch) for batch in loader])
        self.system.fit_standardization(embeddings)

    def _shared_step(self, batch: AudioBatch, stage: str) -> SpeakerGeneratorOutput:
        losses = self.system.compute_loss(self._embeddings(batch))
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
        self.system.save(self.export_path)

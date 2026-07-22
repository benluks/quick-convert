# quick_convert/pipelines/training/sentencepiece_trainer.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .base_trainer import BaseTrainer


class TokenizerTrainer(BaseTrainer):
    def __init__(
        self,
        module: Any,
        model_prefix: str = "tokenizer",
        text_key: str = "transcript",
        output_dir: str | Path | None = None,
    ) -> None:
        super().__init__(output_dir=output_dir)
        self.module = module
        self.model_prefix = model_prefix
        self.text_key = text_key

    def _iter_sentences(self, dataset: Any) -> Iterable[str]:
        for item in dataset:
            try:
                text = item.resources[self.text_key].value
            except (KeyError, AttributeError):
                raise ValueError(f"Each item in the dataset must have a resource with key '{self.text_key}'.")

            if text:
                yield str(text)

    def train(
        self,
        train_dataset: Any | None = None,
        val_dataset: Any | None = None,
        test_dataset: Any | None = None,
        out_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> Path:
        if train_dataset is None:
            raise ValueError(f"{self.__class__.__name__} requires `train_dataset`.")

        resolved_output_dir = self.resolve_output_dir(out_dir)
        if resolved_output_dir is None:
            raise ValueError("No output_dir was provided.")

        return self.module.train_from_iterator(
            sentences=self._iter_sentences(train_dataset),
            output_dir=resolved_output_dir,
            model_prefix=self.model_prefix,
        )

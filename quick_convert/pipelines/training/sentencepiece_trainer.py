# quick_convert/systems/training/tokenizer.py

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
            if hasattr(item, "annotations") and (getattr(item, "annotations").get("transcript", None) is not None):
                text = item.annotations.get("transcript")
            elif isinstance(item, dict):
                text = item[self.text_key]
            elif hasattr(item, "features") and self.text_key in item.features:
                text = item.features[self.text_key]
            else:
                raise KeyError(f"Could not find text field {self.text_key!r} in dataset item.")

            if text:
                yield str(text)

    def train(
        self,
        train_dataset: Any | None = None,
        val_dataset: Any | None = None,
        test_dataset: Any | None = None,
        output_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> Path:
        if train_dataset is None:
            raise ValueError(f"{self.__class__.__name__} requires `train_dataset`.")

        resolved_output_dir = self.resolve_output_dir(output_dir)
        if resolved_output_dir is None:
            raise ValueError("No output_dir was provided.")

        return self.module.train_from_iterator(
            sentences=self._iter_sentences(train_dataset),
            output_dir=resolved_output_dir,
            model_prefix=self.model_prefix,
        )

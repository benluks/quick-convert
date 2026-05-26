from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Union

import sentencepiece as spm

logger = logging.getLogger(__name__)


class SentencePieceBPETrainer:
    """Train a SentencePiece BPE tokenizer from a text corpus.

    Args:
        vocab_size:
            Target vocabulary size (including special tokens).
        character_coverage:
            Fraction of characters to cover. Use ``1.0`` for languages with
            small alphabets (e.g. Latin-script languages); ``0.9995`` is
            recommended for languages with large character sets (e.g. CJK).
        pad_id:
            ID reserved for the padding token (``-1`` to disable).
        unk_id:
            ID reserved for the unknown token.
        bos_id:
            ID reserved for the begin-of-sequence token (``-1`` to disable).
        eos_id:
            ID reserved for the end-of-sequence token (``-1`` to disable).
        user_defined_symbols:
            Extra symbols to add to the vocabulary verbatim (e.g. task tags).
        input_sentence_size:
            Maximum number of sentences sampled from the input during
            training.  ``0`` means unlimited.
        shuffle_input_sentence:
            Whether to shuffle sampled sentences before training.
        num_threads:
            Number of threads used by the SentencePiece trainer.
    """

    MODEL_TYPE = "bpe"

    def __init__(
        self,
        vocab_size: int = 1000,
        character_coverage: float = 1.0,
        pad_id: int = 0,
        unk_id: int = 1,
        bos_id: int = 2,
        eos_id: int = 3,
        user_defined_symbols: Optional[list[str]] = None,
        input_sentence_size: int = 0,
        shuffle_input_sentence: bool = True,
        num_threads: int = 4,
    ) -> None:
        
        self.vocab_size = vocab_size
        self.character_coverage = character_coverage
        self.pad_id = pad_id
        self.unk_id = unk_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.user_defined_symbols = user_defined_symbols or []
        self.input_sentence_size = input_sentence_size
        self.shuffle_input_sentence = shuffle_input_sentence
        self.num_threads = num_threads

        self._model: Optional[spm.SentencePieceProcessor] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_from_iterator(
        self,
        sentences: Iterable[str],
        output_dir: Union[str, Path],
        model_prefix: str = "tokenizer",
    ) -> Path:
        """Train from an in-memory iterable of sentences.

        Args:
            sentences:
                An iterable that yields one sentence (str) at a time.
            output_dir:
                Directory where the model files will be written.
            model_prefix:
                Base name for the output files.

        Returns:
            Path to the saved ``.model`` file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"{model_prefix}.model"

        # SentencePiece can accept a Python iterator via the `sentence_iterator`
        # keyword when `input` is left empty.
        logger.info(
            "Training SentencePiece BPE tokenizer from iterator → %s", model_path
        )
        self._run_trainer(
            sentence_iterator=iter(sentences),
            model_prefix=str(output_dir / model_prefix),
        )
        self._model = self._load(model_path)
        return model_path

    def encode(self, text: str) -> list[int]:
        """Encode *text* to a list of token IDs."""
        if self._model is None:
            raise ValueError("Tokenizer model not loaded. Call train_from_iterator() or load() first.")
        return self._model.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back to a string."""
        if self._model is None:
            raise ValueError("Tokenizer model not loaded. Call train_from_iterator() or load() first.")
        return self._model.decode(ids)

    def load(self, model_path: Union[str, Path]) -> "SentencePieceBPETrainer":
        """Load a previously saved ``.model`` file."""
        self._model = self._load(Path(model_path))
        return self

    def _train(self, model_prefix: str, **extra_kwargs) -> None:
        """Invoke ``sentencepiece.SentencePieceTrainer.train``."""
        kwargs: dict = dict(
            model_prefix=model_prefix,
            model_type=self.MODEL_TYPE,
            vocab_size=self.vocab_size,
            character_coverage=self.character_coverage,
            pad_id=self.pad_id,
            unk_id=self.unk_id,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            user_defined_symbols=self.user_defined_symbols,
            input_sentence_size=self.input_sentence_size,
            shuffle_input_sentence=self.shuffle_input_sentence,
            num_threads=self.num_threads,
        )
        kwargs.update(extra_kwargs)
        spm.SentencePieceTrainer.train(**kwargs)

    @property
    def vocab_size_actual(self) -> int:
        """Actual vocabulary size of the loaded model."""
        if self._model is None:
            raise ValueError("Tokenizer model not loaded. Call train_from_iterator() or load() first.")
        return self._model.get_piece_size()

    @staticmethod
    def _load(model_path: Path) -> spm.SentencePieceProcessor:
        processor = spm.SentencePieceProcessor()
        processor.load(str(model_path))
        return processor


from os import PathLike
from typing import TYPE_CHECKING

import torch

from quick_convert.data.types import AudioBatch
from quick_convert.systems.asr.utils import greedy_ctc_decode


if TYPE_CHECKING:
    from sentencepiece import SentencePieceProcessor


class ASRLoggingMixin:
    tokenizer: "SentencePieceProcessor | None" = None

    _asr_hypotheses: list[str]
    _asr_references: list[str]
    _asr_utterance_ids: list[str]
    _asr_table_max_rows: int

    def setup_asr_logging(
        self,
        tokenizer_model_path: PathLike,
        table_max_rows: int = 100,
    ) -> None:
        try:
            from sentencepiece import SentencePieceProcessor
        except ImportError as error:
            raise ImportError("ASR logging requires the optional `sentencepiece` dependency.") from error

        # Importing this module itself requires jiwer, so keep it lazy too.
        try:
            from quick_convert.pipelines.evaluation.metrics.wer.jiwer_wer import (
                JiwerWER,
            )
        except ImportError as error:
            raise ImportError("ASR logging requires the optional `jiwer` dependency.") from error

        self.tokenizer = SentencePieceProcessor(
            model_file=str(tokenizer_model_path),
        )
        self._asr_wer = JiwerWER()

        self._asr_hypotheses = []
        self._asr_references = []
        self._asr_utterance_ids = []

        if table_max_rows < 1:
            raise ValueError(f"table_max_rows must be positive, got {table_max_rows}.")

        self._asr_table_max_rows = table_max_rows

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()

        self._asr_hypotheses = []
        self._asr_references = []
        self._asr_utterance_ids = []

    def log_asr_validation_output(
        self,
        batch: AudioBatch,
        logits: torch.Tensor,
        batch_idx: int,
    ) -> None:
        """
        Decode and accumulate a batch of CTC predictions.

        Args:
            batch:
                Validation batch containing transcript resources.

            logits:
                CTC logits shaped [T, B, V].

            batch_idx:
                Validation batch index. Decoded examples are logged only
                for the first validation batch.
        """
        if self.tokenizer is None:
            raise RuntimeError("ASR logging has not been initialized. Call setup_asr_logging() first.")

        try:
            transcripts = self.get_resource(batch, "transcript").values
        except RuntimeError:
            # if transcripts aren't passed as a resource, decode the batch's token ids (at least one has to exist to run supervised asr)
            transcripts = self.tokenizer.decode(batch.resources["token_ids"].values.tolist())

        # [T, B, V] -> [B, T, V]
        batch_logits = logits.transpose(0, 1)

        for i, (item, item_logits) in enumerate(zip(batch, batch_logits, strict=True)):
            hypothesis_ids = greedy_ctc_decode(
                logits=item_logits,
            )

            hypothesis = self.tokenizer.decode_ids(
                hypothesis_ids.tolist(),
            )

            reference = transcripts[i]

            self._asr_hypotheses.append(hypothesis)
            self._asr_references.append(reference)
            self._asr_utterance_ids.append(item.utt_id)

            if batch_idx == 0:
                self._log_asr_example(
                    utt_id=item.utt_id,
                    hypothesis=hypothesis,
                    reference=reference,
                )

    def _log_asr_example(
        self,
        utt_id: str,
        hypothesis: str,
        reference: str,
    ) -> None:
        self.media_logger.log_text(
            {
                f"transcript/{utt_id}/hypothesis": hypothesis,
                f"transcript/{utt_id}/ground_truth": reference,
            },
            step=self.global_step,
        )

    def on_validation_epoch_end(self) -> None:
        if self._asr_references:
            wer = self._asr_wer.compute(
                self._asr_references,
                self._asr_hypotheses,
            )["wer"]

            self.log(
                "val/wer",
                wer,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

            self._log_asr_table()

        super().on_validation_epoch_end()

    def _log_asr_table(self) -> None:
        trainer = getattr(self, "trainer", None)

        if trainer is not None and not trainer.is_global_zero:
            return

        try:
            from lightning.pytorch.loggers import WandbLogger
        except ImportError:
            return

        loggers = getattr(self, "loggers", None)

        if loggers is None:
            logger = getattr(self, "logger", None)
            loggers = [] if logger is None else [logger]

        rows = list(
            zip(
                self._asr_utterance_ids[: self._asr_table_max_rows],
                self._asr_references[: self._asr_table_max_rows],
                self._asr_hypotheses[: self._asr_table_max_rows],
                strict=True,
            )
        )

        for logger in loggers:
            if isinstance(logger, WandbLogger):
                logger.log_table(
                    key="val/asr_transcripts",
                    columns=["utterance_id", "reference", "hypothesis"],
                    data=rows,
                    step=self.global_step,
                )

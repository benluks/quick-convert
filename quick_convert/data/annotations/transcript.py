# quick_convert/data/annotations/transcripts.py

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .base import BaseAnnotationProvider, PathFormatter


class CSVTranscriptProvider(BaseAnnotationProvider):
    def __init__(
        self,
        name: str = "transcript",
        path_template: str | None = None,
        transcript_path_key: str | None = "transcript_path",
        utterance_key: str = "path.stem",
        key_column: int = 0,
        text_column: int = 1,
        delimiter: str | None = None,
        encoding: str = "utf-8",
        join_text_columns: bool = False,
    ) -> None:
        super().__init__(name=name)

        self.path_template = path_template
        self.transcript_path_key = transcript_path_key
        self.utterance_key = utterance_key
        self.key_column = key_column
        self.text_column = text_column
        self.delimiter = delimiter
        self.encoding = encoding
        self.join_text_columns = join_text_columns

        self._cache: dict[Path, dict[str, str]] = {}

    def __call__(self, sample: Any) -> str:
        transcript_path = self._resolve_transcript_path(sample)

        if transcript_path not in self._cache:
            self._cache[transcript_path] = self._load_transcript_file(transcript_path)

        utterance_id = str(self._get_sample_value(sample, self.utterance_key))

        try:
            return self._cache[transcript_path][utterance_id]
        except KeyError as e:
            raise KeyError(
                f"No transcript found for utterance_id={utterance_id!r} in transcript file {transcript_path}"
            ) from e

    def _resolve_transcript_path(self, sample: Any) -> Path:
        if self.path_template is not None:
            return PathFormatter.format(sample, self.path_template)

        if self.transcript_path_key is None:
            raise ValueError("Either path_template or transcript_path_key must be provided.")

        rel_path = Path(self._get_sample_value(sample, self.transcript_path_key))
        audio_path = Path(self._get_sample_value(sample, "path"))

        return (audio_path.parent / rel_path).resolve()

    def _load_transcript_file(self, path: Path) -> dict[str, str]:
        if not path.exists():
            raise FileNotFoundError(f"Transcript file not found: {path}")

        index: dict[str, str] = {}

        with path.open("r", encoding=self.encoding, newline="") as f:
            reader = csv.reader(f, delimiter=self.delimiter) if self.delimiter else csv.reader(f)

            for row_number, row in enumerate(reader, start=1):
                if len(row) <= max(self.key_column, self.text_column):
                    raise ValueError(
                        f"Row {row_number} in {path} has only {len(row)} columns, "
                        f"but key_column={self.key_column} and text_column={self.text_column}"
                    )

                key = row[self.key_column].strip()
                if self.join_text_columns:
                    text = " ".join(row[self.text_column :]).strip()
                else:
                    text = row[self.text_column].strip()

                if key in index:
                    raise ValueError(f"Duplicate transcript key {key!r} in {path}")

                index[key] = text

        return index

    def _get_sample_value(self, sample: Any, key: str) -> Any:
        return PathFormatter._get_sample_value(sample, key)

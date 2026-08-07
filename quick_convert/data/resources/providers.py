import csv
from pathlib import Path
from typing import Any

import torch

from ...utils.paths import SamplePathFormatter
from ..types import AudioBatch, AudioSample
from .base import ResourceRef


class BaseResourceProvider:
    """
    An abstracton class for resource providers, which are responsible for providing access to various types of
    resources (e.g. annotation files, precompute feature files, etc.) associated with samples in a dataset.
    """

    def __init__(self, name: str):
        self.name = name

    def __call__(self, sample):
        raise NotImplementedError


class TemplateResourceProvider(BaseResourceProvider):
    def __init__(self, name: str, template: str, kind: str = "text"):
        super().__init__(name)
        self.template = template
        self.kind = kind

    def resolve(self, sample):
        return SamplePathFormatter.format_str(sample, self.template)

    def __call__(self, sample):
        return ResourceRef(name=self.name, kind=self.kind, value=self.resolve(sample))


class PathResourceProvider(TemplateResourceProvider):
    def __init__(
        self,
        name,
        path_template,
        kind,
        # only set `max_length` for tensor resources
        max_length=None,
        must_exist=True,
    ):
        super().__init__(name=name, template=path_template, kind=kind)
        self.max_length = max_length
        self.must_exist = must_exist

    def __call__(self, sample):
        path = Path(self.resolve(sample))

        if self.must_exist and not path.exists():
            raise FileNotFoundError(f"Missing resource {self.name}: {path}")

        return ResourceRef(name=self.name, kind=self.kind, path=path, value=None, max_length=self.max_length)


class CSVAnnotationProvider(BaseResourceProvider):
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
            transcript = self._cache[transcript_path][utterance_id]
            return ResourceRef(value=transcript, kind="text", name=self.name, path=transcript_path)
        except KeyError as e:
            raise KeyError(
                f"No transcript found for utterance_id={utterance_id!r} in transcript file {transcript_path}"
            ) from e

    def _resolve_transcript_path(self, sample: Any) -> Path:
        if self.path_template is not None:
            return SamplePathFormatter.format(sample, self.path_template)

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
        return SamplePathFormatter._get_sample_value(sample, key)


class OnlineResourceProvider:
    def __init__(
        self,
        extractor: Any,
        name: str | None = None,
    ):
        self.extractor = extractor
        self.name = extractor.feature_name if name is None else name

    def provide_sample(self, sample: AudioSample) -> torch.Tensor:
        return self.extractor.extract_sample(sample)

    def provide_batch(self, batch: AudioBatch) -> torch.Tensor:
        return self.extractor.extract_batch(batch)

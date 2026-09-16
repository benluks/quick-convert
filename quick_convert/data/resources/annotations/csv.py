from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ....utils.paths import SamplePathFormatter
from ..base import BaseResourceProvider, ResourceRef


class CSVAnnotationProvider(BaseResourceProvider):
    """Resolve arbitrary text annotations from a shared delimited file."""

    def __init__(
        self,
        name: str,
        path_template: str | None = None,
        annotation_path_key: str | None = "annotation_path",
        item_key: str = "path.stem",
        key_column: int = 0,
        value_column: int = 1,
        delimiter: str | None = None,
        encoding: str = "utf-8",
        join_value_columns: bool = False,
    ) -> None:
        super().__init__(name=name)

        self.path_template = path_template
        self.annotation_path_key = annotation_path_key
        self.item_key = item_key
        self.key_column = key_column
        self.value_column = value_column
        self.delimiter = delimiter
        self.encoding = encoding
        self.join_value_columns = join_value_columns

        self._cache: dict[Path, dict[str, str]] = {}

    def __call__(self, sample: Any) -> ResourceRef:
        annotation_path = self._resolve_annotation_path(sample)

        if annotation_path not in self._cache:
            self._cache[annotation_path] = self._load_annotation_file(annotation_path)

        item_id = str(self._get_sample_value(sample, self.item_key))

        try:
            value = self._cache[annotation_path][item_id]
        except KeyError as error:
            raise KeyError(f"No annotation found for item {item_id!r} in {annotation_path}") from error

        return ResourceRef(value=value, kind="text", name=self.name, path=annotation_path)

    def _resolve_annotation_path(self, sample: Any) -> Path:
        if self.path_template is not None:
            return SamplePathFormatter.format(sample, self.path_template)

        if self.annotation_path_key is None:
            raise ValueError("Either path_template or annotation_path_key must be provided.")

        relative_path = Path(self._get_sample_value(sample, self.annotation_path_key))
        sample_path = Path(self._get_sample_value(sample, "path"))
        return (sample_path.parent / relative_path).resolve()

    def _load_annotation_file(self, path: Path) -> dict[str, str]:
        if not path.exists():
            raise FileNotFoundError(f"Annotation file not found: {path}")

        index: dict[str, str] = {}
        with path.open("r", encoding=self.encoding, newline="") as annotation_file:
            reader = (
                csv.reader(annotation_file, delimiter=self.delimiter) if self.delimiter else csv.reader(annotation_file)
            )

            for row_number, row in enumerate(reader, start=1):
                if len(row) <= max(self.key_column, self.value_column):
                    raise ValueError(
                        f"Row {row_number} in {path} has only {len(row)} columns, "
                        f"but key_column={self.key_column} and value_column={self.value_column}"
                    )

                key = row[self.key_column].strip()
                value = (
                    " ".join(row[self.value_column :]).strip()
                    if self.join_value_columns
                    else row[self.value_column].strip()
                )

                if key in index:
                    raise ValueError(f"Duplicate annotation key {key!r} in {path}")

                index[key] = value

        return index

    @staticmethod
    def _get_sample_value(sample: Any, key: str) -> Any:
        return SamplePathFormatter._get_sample_value(sample, key)

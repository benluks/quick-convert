"""Resource providers for attaching data to dataset samples.

Providers separate dataset membership from experiment-specific data. A
``BaseDataset`` determines which utterances exist; providers determine which
additional resources belong to each utterance.

Reference-based providers return :class:`ResourceRef` objects. Online providers
compute values dynamically from an ``AudioSample`` or ``AudioBatch``.
"""

import csv
from pathlib import Path
from typing import Any

import torch

from ...utils.paths import SamplePathFormatter
from .base import ResourceRef


class BaseResourceProvider:
    """Base interface for sample-level resource providers.

    A provider maps a sample to a named :class:`ResourceRef`. Providers should
    resolve *where a resource comes from* without deciding whether a
    path-backed resource should be loaded; loading is controlled by the
    dataset's ``load`` policy.

    Subclasses implement :meth:`__call__`.
    """

    def __init__(self, name: str):
        self.name = name

    def __call__(self, sample) -> ResourceRef:
        """Return the resource associated with ``sample``."""
        raise NotImplementedError

    def __init__(self, name: str):
        self.name = name

    def __call__(self, sample):
        raise NotImplementedError


class TemplateResourceProvider(BaseResourceProvider):
    """Provide an in-memory resource by formatting sample metadata.

    This is useful for lightweight metadata that can be derived directly from
    the sample, such as speaker IDs, session names, language labels, or other
    path-derived values.

    Example:
        Derive the LibriSpeech speaker ID from its directory structure::

            provider = TemplateResourceProvider(
                name="speaker_id",
                template="{path.parent.parent.name}",
                kind="text",
            )

    Args:
        name:
            Resource name exposed on the sample.
        template:
            Template evaluated against the sample.
        kind:
            Resource kind assigned to the resulting reference.
    """

    def __init__(self, name: str, template: str, kind: str = "text"):
        super().__init__(name)
        self.template = template
        self.kind = kind

    def resolve(self, sample):
        return SamplePathFormatter.format_str(sample, self.template)

    def __call__(self, sample):
        return ResourceRef(name=self.name, kind=self.kind, value=self.resolve(sample))


class PathResourceProvider(TemplateResourceProvider):
    """Associate each sample with a path-backed resource.

    The path is produced by formatting ``path_template`` against the sample.
    The returned :class:`ResourceRef` remains unresolved until the dataset's
    ``load`` policy requests the resource.

    Example:
        Associate each utterance with a precomputed WavLM tensor::

            provider = PathResourceProvider(
                name="wavlm",
                path_template="/features/wavlm/{sample.utt_id}.pt",
                kind="torch_tensor",
            )

    Args:
        name:
            Resource name exposed on the sample.
        path_template:
            Template resolving to the resource path.
        kind:
            Resource kind used for loading and collation.
        max_length:
            Optional fixed padding length for tensor resources.
        must_exist:
            Whether to raise immediately if the resolved path does not exist.
    """

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
    """Look up per-utterance text annotations from shared delimited files.

    The provider resolves an annotation file for each sample, parses each file
    once, and caches an ``utterance_id -> text`` lookup table for subsequent
    samples.

    This is useful for corpora such as LibriSpeech where many utterances share
    a transcript file.

    The annotation file can be located either with ``path_template`` or from a
    path stored on the sample via ``transcript_path_key``.

    Args:
        name:
            Resource name. Defaults to ``"transcript"``.
        path_template:
            Optional template resolving directly to the annotation file.
        transcript_path_key:
            Sample field containing a transcript path when
            ``path_template`` is not supplied.
        utterance_key:
            Sample expression used to determine the lookup key.
        key_column:
            Column containing utterance IDs.
        text_column:
            First column containing annotation text.
        delimiter:
            Optional CSV delimiter.
        encoding:
            Text encoding used to read annotation files.
        join_text_columns:
            If true, join all columns beginning at ``text_column``.

    Raises:
        FileNotFoundError:
            If a resolved annotation file does not exist.
        KeyError:
            If the current sample has no annotation in the resolved file.
        ValueError:
            If an annotation file is malformed or contains duplicate keys.
    """

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
    """Compute a resource dynamically using a feature extractor.

    Online providers are intended for resources that should be computed during
    an experiment rather than loaded from precomputed sidecar files. The
    wrapped extractor must provide ``extract_sample`` and ``extract_batch``
    methods.

    If ``name`` is omitted, ``extractor.feature_name`` is used.

    Example::

        provider = OnlineResourceProvider(
            extractor=wavlm_encoder,
            name="wavlm",
        )

        frame_features = provider.provide_batch(batch)

    Args:
        extractor:
            Feature extractor used to compute the resource.
        name:
            Optional resource name.
    """

    def __init__(
        self,
        extractor: Any,
        name: str | None = None,
    ):
        self.extractor = extractor
        self.name = extractor.feature_name if name is None else name

    def provide_sample(self, sample) -> torch.Tensor:
        """Compute the resource for one sample."""
        return self.extractor.extract_sample(sample)

    def provide_batch(self, batch) -> torch.Tensor:
        """Compute the resource for a batch."""
        return self.extractor.extract_batch(batch)

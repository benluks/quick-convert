from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from fnmatch import fnmatch
from os import PathLike
from pathlib import Path
from typing import Any, Literal

from torch.utils.data import DataLoader, Dataset

from quick_convert.utils.paths import TemplateFormatter

from ..utils.audio import get_supported_formats, load_audio
from .resources import BaseResourceProvider, ResourceCollection, ResourceRef, load_resource
from .types import AudioBatch, AudioSample, MetadataBatch, MetadataSample


class BaseDataset(Dataset):
    """Generic audio dataset with composable resources.

    ``BaseDataset`` discovers audio samples from a directory, an explicit list
    of paths, or preconstructed :class:`MetadataSample` rows. Additional data
    associated with each sample—such as transcripts, speaker IDs, SSL
    features, token sequences, or acoustic measurements—is attached through
    resource providers rather than dataset-specific fields.

    Audio and path-backed resources are loaded lazily according to ``load``.
    This makes it possible to use the same dataset definition for lightweight
    metadata inspection, preprocessing, and model training.

    Exactly one of ``root``, ``paths``, or ``rows`` must be provided.

    Args:
        root:
            Root directory containing audio files. If ``splits`` is provided,
            each split is interpreted as a subdirectory of ``root``.
        splits:
            Optional split directories to include, for example
            ``["train-clean-100", "train-clean-360"]``.
        file_format:
            Audio format or formats to include, such as ``"flac"`` or
            ``["wav", "flac"]``. If omitted, all supported audio formats are
            accepted.
        paths:
            Explicit audio paths from which to construct the dataset.
        rows:
            Preconstructed sample metadata. This is primarily useful for
            dataset subclasses such as :class:`ManifestDataset`.
        load:
            Controls which resources are materialized when a sample is
            accessed.

            - ``False`` or ``None`` loads no audio or path-backed resources.
            - A list such as ``["audio", "wavlm"]`` loads only those names.
            - ``True`` or ``"all"`` loads audio and all configured resources.

            Resources that already contain an in-memory value do not require
            loading and are available regardless of this setting.
        target_sr:
            Optional sampling rate used when loading audio. Audio is resampled
            when necessary.
        convert_to_mono:
            Whether loaded audio should be converted to mono.
        utt_id_template:
            Template used to derive utterance IDs from discovered paths, for
            example ``"{path.stem}"``.
        get_utt_id_fn:
            Optional callable used instead of ``utt_id_template`` to derive an
            utterance ID from a path.
        pattern:
            Glob pattern used during recursive file discovery. Defaults to
            ``"*"``.
        exclude_patterns:
            Optional filename or path patterns to exclude.
        resource_providers:
            Providers evaluated for every sample. Each provider associates a
            named resource with the sample.
        sort_key:
            Template used to sort discovered rows. Defaults to ``"{row.path}"``.
        max_length:
            Optional waveform length to pad batches to, expressed in samples
            after resampling. Useful when fixed input shapes are required.

    Examples:
        Create a simple filesystem dataset::

            dataset = BaseDataset(
                root="/data/LibriSpeech",
                splits=["train-clean-100"],
                file_format="flac",
                utt_id_template="{path.stem}",
                load=["audio"],
                target_sr=16_000,
            )

        Add precomputed features without changing the dataset class::

            wavlm = PathResourceProvider(
                name="wavlm",
                path_template="/features/wavlm/{sample.utt_id}.pt",
                kind="torch_tensor",
            )

            dataset = BaseDataset(
                root="/data/LibriSpeech",
                splits=["train-clean-100"],
                file_format="flac",
                utt_id_template="{path.stem}",
                resource_providers=[wavlm],
                load=["audio", "wavlm"],
            )

            sample = dataset[0]
            sample.resources.wavlm.value
    """

    VALID_FORMATS = get_supported_formats()

    def __init__(
        self,
        root: str | Path | None = None,
        splits: Iterable[str] | None = None,
        file_format: str | Iterable[str] | None = None,
        paths: Iterable[str | Path] | None = None,
        rows: Iterable[MetadataSample] | None = None,
        load: bool | list[str] | Literal["all"] | None = False,
        target_sr: int | None = None,
        convert_to_mono: bool = True,
        utt_id_template: str | None = None,
        get_utt_id_fn: Callable[[PathLike], str] | None = None,
        pattern: str | None = None,
        exclude_patterns: Iterable[str] | None = None,
        resource_providers: Iterable[BaseResourceProvider] | None = None,
        sort_key: str | None = "{row.path}",
        # length to extend collated audio files to beyond the maximum sample length. This is used in
        # cudnn benchmark where all batches must have the same shape. Expressed in number of samples after resampling
        max_length: int | None = None,
        **kwargs,
    ):
        # Samples may come from filesystem discovery, explicit paths, or an
        # already-constructed metadata table. Mixing sources would make row
        # ownership and filtering ambiguous.
        sources = [
            root is not None,
            paths is not None,
            rows is not None,
        ]

        if sum(sources) != 1:
            raise ValueError(
                f"Provide exactly one of `root`, `paths`, or `rows`. Got root={root}, paths={paths}, rows={rows}."
            )

        self.file_formats = self._normalize_and_validate_format(file_format)
        self.splits = list(splits) if splits is not None else None
        self.convert_to_mono = convert_to_mono

        # for determining utterance ID
        self.utt_id_template = utt_id_template
        self.get_utt_id_fn = get_utt_id_fn

        self.target_sr = target_sr
        self.root = Path(root) if root is not None else None

        self.pattern = pattern or "*"
        self.exclude_patterns = exclude_patterns or []
        self.resource_providers = resource_providers or []

        # Normalize once so __getitem__ only needs a cheap membership check.
        self.load = self._normalize_load(load)
        self.max_length = max_length

        if rows is not None:
            self.rows = rows
            return

        elif paths is not None:
            rows = []
            files = [Path(p) for p in paths if Path(p).is_file()]
            for p in files:
                rows.append(
                    MetadataSample(
                        utt_id=self.get_utt_id(p),
                        path=p,
                    )
                )
        else:
            if not self.root.exists():
                raise FileNotFoundError(f"Directory does not exist: {self.root}")
            if not self.root.is_dir():
                raise NotADirectoryError(f"Expected a directory: {self.root}")

            if self.splits is None:
                search_roots = [(None, self.root)]
            else:
                search_roots = []
                for split in self.splits:
                    split_root = self.root / split
                    if not split_root.exists():
                        raise FileNotFoundError(f"Split directory does not exist: {split_root}")
                    if not split_root.is_dir():
                        raise NotADirectoryError(f"Expected a directory: {split_root}")
                    search_roots.append((split, split_root))

            file_formats = self.file_formats if self.file_formats is not None else self.VALID_FORMATS
            rows = []
            for split, search_root in search_roots:
                for p in search_root.rglob(self.pattern):
                    if not p.is_file():
                        continue
                    if self._is_excluded(p):
                        continue
                    if p.suffix.lower().lstrip(".") not in file_formats:
                        continue
                    rows.append(
                        MetadataSample(
                            utt_id=self.get_utt_id(p),
                            path=p,
                            split=split,
                        )
                    )

        self.sort_key = sort_key
        self.rows = sorted(rows, key=lambda row: TemplateFormatter.format_str(sort_key, row=row))

    @classmethod
    def _normalize_and_validate_format(cls, file_format: str | Iterable[str] | None) -> set[str] | None:
        """Normalize requested audio extensions and reject unsupported formats."""
        if file_format is None:
            return None

        if isinstance(file_format, str):
            formats = [file_format]
        else:
            formats = list(file_format)

        normalized = set()
        for fmt in formats:
            fmt = fmt.lower().strip().lstrip(".")
            if fmt not in cls.VALID_FORMATS:
                valid = ", ".join(sorted(cls.VALID_FORMATS))
                raise ValueError(f"Invalid audio format: {fmt!r}. Valid formats are: {valid}")
            normalized.add(fmt)

        return normalized

    def _normalize_load(self, load: bool | list[str] | Literal["all"] | None) -> bool | set[str]:
        """Convert the public ``load`` argument into a set of resource names."""
        if load is None or load is False:
            return []

        if load is True or load == "all":
            return {"audio"}.union({provider.name for provider in self.resource_providers})

        if isinstance(load, str):
            return {load}

        return set(load)

    def _should_load(self, ref: ResourceRef | Literal["audio"]) -> bool:
        """Return whether an unresolved resource should be materialized.

        Resources that already contain a value are never loaded again.
        """
        if getattr(ref, "value", None) is not None:
            return False
        name = "audio" if ref == "audio" else ref.name
        return name in self.load

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> AudioSample:
        sample = self.rows[idx]

        if self._should_load("audio"):
            sample = self.load_sample(sample)

        # Providers contribute ResourceRefs at access time, allowing the same
        # dataset rows to be reused with different experiment-specific data.
        resource_refs = [
            *sample.resources,
            *(provider(sample) for provider in self.resource_providers),
        ]
        resources = ResourceCollection.from_refs(resource_refs)

        for name, ref in resources.items():
            if self._should_load(ref):
                resources[name] = load_resource(ref)

        # materialize resources here

        return replace(sample, resources=resources)

    def _is_excluded(self, path: Path) -> bool:
        return any(fnmatch(path.name, pattern) or fnmatch(str(path), pattern) for pattern in self.exclude_patterns)

    def get_utt_id(self, path: Path) -> str:
        if self.get_utt_id_fn is not None:
            return self.get_utt_id_fn(path)

        elif self.utt_id_template is not None:
            return TemplateFormatter.format_str(
                self.utt_id_template,
                path=path,
            )
        else:
            raise RuntimeError(
                f"No method for determining utt_id. Please provide either `utt_id_template` or `get_utt_id_fn` when initializing {type(self).__name__}."
            )

    def load_sample(self, sample: AudioSample) -> dict[str, Any]:
        """Load and optionally resample the audio associated with ``sample``.

        The returned sample preserves its metadata and resources while adding
        ``waveform`` and ``sample_rate``.
        """
        waveform, sample_rate = load_audio(sample.path, target_sr=self.target_sr, mono=self.convert_to_mono)
        return AudioSample(
            utt_id=sample.utt_id,
            path=sample.path,
            split=sample.split,
            # spk_id=sample.spk_id,
            waveform=waveform,
            sample_rate=sample_rate,
            resources=sample.resources,
        )

    def collate_fn(self, batch: list[AudioSample]) -> MetadataBatch | AudioBatch:
        return AudioBatch.from_samples(batch, max_length=self.max_length)

    def make_dataloader(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        **kwargs,
    ) -> DataLoader:
        """Create a DataLoader using the dataset's resource-aware collator.

        This is preferred over constructing ``torch.utils.data.DataLoader``
        directly because :meth:`collate_fn` handles variable-length audio and
        resource collation.

        Args:
            batch_size:
                Number of samples per batch.
            shuffle:
                Whether to reshuffle samples each epoch.
            num_workers:
                Number of worker processes used for loading.
            pin_memory:
                Whether DataLoader should place tensors in pinned CPU memory.
            drop_last:
                Whether to discard an incomplete final batch.
            **kwargs:
                Additional arguments forwarded to :class:`DataLoader`.
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=self.collate_fn,
            **kwargs,
        )

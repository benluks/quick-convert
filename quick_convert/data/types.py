"""Sample and batch containers used by the quick-convert data pipeline."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from quick_convert.utils.audio import load_audio

from .resources import ResourceCollection, TemplateResourceProvider, collate_resources, load_resource


@dataclass(frozen=True)
class MetadataSample:
    """Metadata and resources describing one dataset item.

    ``MetadataSample`` does not imply that audio has been loaded. Dataset
    subclasses can construct these rows cheaply and allow ``BaseDataset`` to
    materialize audio and resources later.
    """

    utt_id: str
    path: Path
    split: str | None = None
    resources: ResourceCollection = field(default_factory=ResourceCollection)


@dataclass(frozen=True)
class AudioSample(MetadataSample):
    """A sample with optionally materialized audio.

    ``waveform`` and ``sample_rate`` remain ``None`` until audio is loaded.
    Because samples are immutable, :meth:`load_audio` returns a new sample.
    """

    waveform: float["1 t"] | None = None
    sample_rate: int | None = None

    @classmethod
    def from_path(cls, path: str | Path, utt_id: str | None = None, **kwargs):
        """Construct an unloaded sample from an audio path.

        If ``utt_id`` is omitted, the filename stem is used.
        """
        path = Path(path)
        return cls(
            utt_id=utt_id or path.stem,
            path=path,
            **kwargs,
        )

    def load_audio(self, *args, **kwargs) -> "AudioSample":
        """Return a copy of the sample with its waveform loaded."""
        waveform, sr = load_audio(self.path, *args, **kwargs)
        return replace(self, waveform=waveform, sample_rate=sr)


@dataclass
class MetadataBatch:
    utt_ids: list[str]
    paths: list[Path]
    splits: list[str | None]
    resources: dict[str, Any]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> AudioSample:
        return AudioSample(
            utt_id=self.utt_ids[idx],
            path=self.paths[idx],
            split=self.splits[idx],
            resources={key: value[idx] for key, value in self.resources.items()},
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


@dataclass
class AudioBatch(MetadataBatch):
    """Batch of samples with optionally loaded, padded audio.

    When audio is present, ``waveforms`` has shape ``[B, T]`` and ``lengths``
    stores the unpadded length of each waveform. If the source samples did not
    load audio, all audio-specific fields remain ``None``.
    """

    waveforms: float["b t"] | None = None
    lengths: int["b"] | None = None
    sample_rates: int["b"] | None = None

    @classmethod
    def from_samples(cls, samples: list[AudioSample], max_length: int | None = None) -> "AudioBatch":
        """Collate samples into a batch.

        Audio is padded along time, and named resources are independently
        collated according to their resource kind.

        Args:
            samples:
                Samples to collate.
            max_length:
                Optional fixed waveform length to pad to. It must not be
                shorter than any waveform in the batch.
        """
        has_audio = all(s.waveform is not None for s in samples)

        common_kwargs = {
            "utt_ids": [s.utt_id for s in samples],
            "paths": [s.path for s in samples],
            "splits": [s.split for s in samples],
            "resources": collate_resources(samples),
        }

        if not has_audio:
            return cls(**common_kwargs)

        waveforms = [s.waveform.squeeze(0) for s in samples]
        lengths = torch.tensor([w.shape[-1] for w in waveforms], dtype=torch.long)

        if max_length is not None:
            if (lengths > max_length).any():
                raise ValueError("max_length is shorter than at least one waveform.")
            waveforms[0] = F.pad(waveforms[0], (0, max_length - waveforms[0].shape[-1]))

        return cls(
            **common_kwargs,
            waveforms=pad_sequence(waveforms, batch_first=True),
            lengths=lengths,
            sample_rates=torch.tensor([s.sample_rate for s in samples], dtype=torch.long),
        )

    @classmethod
    def from_paths(
        cls,
        paths: str | Path | list[str | Path],
        resource_providers: Iterable[TemplateResourceProvider] | None = [],
        target_sr: int | None = None,
        mono: bool = True,
        max_length: int | None = None,
        utt_id_fn: Callable[[Path], str] | None = None,
        **kwargs,
    ) -> "AudioBatch":
        """Load audio paths directly into a batch.

        This convenience constructor is useful for inference and ad-hoc feature
        extraction when constructing a full dataset would be unnecessary.

        Optional resource providers may attach additional resources before
        batching.
        """
        if isinstance(paths, (str, Path)):
            paths = [paths]

        samples = []

        for path in map(Path, paths):
            sample = AudioSample.from_path(
                path,
                utt_id=utt_id_fn(path) if utt_id_fn else path.stem,
            )

            if resource_providers:
                resources = ResourceCollection.from_refs([provider(sample) for provider in resource_providers])
                sample = replace(
                    sample,
                    resources=resources,
                )
            for name, ref in resources.items():
                resources[name] = load_resource(ref)
            sample = sample.load_audio(target_sr=target_sr, mono=mono)

            samples.append(sample)

        return cls.from_samples(samples, max_length=max_length)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> AudioSample:
        return AudioSample(
            utt_id=self.utt_ids[idx],
            path=self.paths[idx],
            split=self.splits[idx],
            waveform=self.waveforms[idx] if self.waveforms is not None else None,
            sample_rate=self.sample_rates[idx] if self.sample_rates is not None else None,
            # features={key: value[idx] for key, value in self.features.items()},
            resources={key: value[idx] for key, value in self.resources.items()},
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

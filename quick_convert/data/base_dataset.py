from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Callable, Iterable, Optional, Union, Any

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from ..utils.audio import get_supported_formats


@dataclass(frozen=True)
class MetadataSample:
    path: Path
    split: Optional[str] = None
    spk_id: Optional[str] = None


@dataclass(frozen=True)
class LoadedSample(MetadataSample):
    waveform: float["1 t"]
    sample_rate: int


@dataclass(frozen=True)
class AudioSample(MetadataSample):
    waveform: Optional[float["1 t"]] = None
    sample_rate: Optional[int] = None


@dataclass(frozen=True)
class MetadataBatch:
    paths: list[Path]
    splits: list[str | None]
    spk_ids: list[str | None]


@dataclass(frozen=True)
class LoadedBatch(MetadataBatch):
    waveforms: float["b t"]
    lengths: int["b"]
    sample_rates: int["b"]


@dataclass(frozen=True)
class AudioBatch(MetadataBatch):
    waveforms: Optional[float["b t"]] = None
    lengths: Optional[int["b"]] = None
    sample_rates: Optional[int["b"]] = None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> AudioSample:
        return AudioSample(
            path=self.paths[idx],
            split=self.splits[idx],
            spk_id=self.spk_ids[idx],
            waveform=self.waveforms[idx] if self.waveforms is not None else None,
            sample_rate=self.sample_rates[idx] if self.sample_rates is not None else None,
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class BaseDataset(Dataset):
    VALID_FORMATS = get_supported_formats()

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        splits: Optional[Iterable[str]] = None,
        file_format: Optional[Union[str, Iterable[str]]] = None,
        paths: Optional[Iterable[Union[str, Path]]] = None,
        load: bool = False,
        return_spkid: bool = False,
        target_sr: Optional[int] = None,
        convert_to_mono: bool = True,
        # pass a spkid function to avoid subclassing just to implement get_spkid logic
        get_spkid_fn: Optional[Callable[[PathLike], str]] = None,
    ):
        if root is None and paths is None:
            raise ValueError("You must provide either `root` or `paths`.")
        if root is not None and paths is not None:
            raise ValueError("Provide only one of `root` or `paths`, not both.")

        self.file_formats = self._normalize_and_validate_format(file_format)
        self.splits = list(splits) if splits is not None else None
        self.convert_to_mono = convert_to_mono
        self.target_sr = target_sr
        self.root = Path(root) if root is not None else None
        self.load = load
        self.return_spkid = return_spkid
        if get_spkid_fn is not None:
            self.get_spkid = get_spkid_fn

        rows: list[MetadataSample] = []

        if paths is not None:
            files = [Path(p) for p in paths if Path(p).is_file()]
            for p in files:
                rows.append(
                    MetadataSample(
                        path=p,
                        spk_id=self.get_spkid(p) if return_spkid else None,
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

            for split, search_root in search_roots:
                for p in search_root.rglob("*"):
                    if not p.is_file():
                        continue
                    if p.suffix.lower().lstrip(".") not in file_formats:
                        continue
                    rows.append(
                        MetadataSample(
                            path=p,
                            split=split,
                            spk_id=self.get_spkid(p) if return_spkid else None,
                        )
                    )

        self.rows = sorted(rows, key=lambda row: str(row.path))

    @classmethod
    def _normalize_and_validate_format(cls, file_format: Optional[Union[str, Iterable[str]]]) -> Optional[set[str]]:
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

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> AudioSample | dict[str, Any]:
        sample = self.rows[idx]
        if self.load:
            return self.load_sample(sample)
        return sample

    def get_spkid(self, file_path: PathLike) -> str:
        raise NotImplementedError(f"{type(self).__name__} must implement `get_spkid` when `return_spkid=True`.")

    def load_audio(self, path: Path, sample_rate: Optional[int] = None) -> tuple[torch.Tensor, int]:
        waveform, sr = torchaudio.load(str(path))
        # Convert to mono if needed.
        if waveform.dim() == 2 and waveform.shape[0] > 1 and self.convert_to_mono:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate is not None:
            torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=sample_rate)
            sr = sample_rate
        return waveform, sr

    def load_sample(self, sample: AudioSample) -> dict[str, Any]:
        """
        If target_sr is set, loading will resample audio. I haven't implemented a way to override this.
        Maybe it's better to leave the resampling concern to a different part of the pipeline. Time will tell.
        """
        waveform, sample_rate = self.load_audio(sample.path, self.target_sr)
        return AudioSample(
            path=sample.path,
            split=sample.split,
            spk_id=sample.spk_id,
            waveform=waveform,
            sample_rate=sample_rate,
        )

    def collate_fn(self, batch: list[AudioSample]) -> Any:
        """
        Default collate behavior.

        - If self.load=False, returns a metadata batch.
        - If self.load=True, pads variable-length waveforms and returns tensors + metadata.
        """
        if not self.load:
            return MetadataBatch(
                paths=[item.path for item in batch],
                splits=[item.split for item in batch],
                spk_ids=[item.spk_id for item in batch],
            )

        # list[[1 t]]
        waveforms = [item.waveform.squeeze(0) for item in batch]
        lengths = torch.tensor([w.shape[-1] for w in waveforms], dtype=torch.long)

        padded = pad_sequence(waveforms, batch_first=True)

        sample_rates = [item["sample_rate"] for item in batch]
        # if len(set(sample_rates)) != 1:
        #     raise ValueError(f"Batch contains multiple sample rates: {sorted(set(sample_rates))}")

        return AudioBatch(
            waveforms=padded,
            lengths=lengths,
            sample_rates=sample_rates,
            paths=[item["path"] for item in batch],
            splits=[item["split"] for item in batch],
            spk_ids=[item["spk_id"] for item in batch],
        )

    def make_dataloader(
        self,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        **kwargs,
    ) -> DataLoader:
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

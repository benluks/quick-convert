import logging
from pathlib import Path
from typing import Iterable, Optional, Union

from torch.utils.data import Dataset

from ..utils.audio import get_supported_formats

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioSample:
    path: Path
    split: str | None = None
    spk_id: Optional[str] = None


class BaseDataset(Dataset):
    VALID_FORMATS = get_supported_formats()

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        splits: Optional[Iterable[str]] = None,
        file_format: Optional[Union[str, Iterable[str]]] = None,
        paths: Optional[Iterable[Union[str, Path]]] = None,
        load: bool = False,
        return_spkid=False,
    ):
        if root is None and paths is None:
            raise ValueError("You must provide either `root` or `paths`.")
        if root is not None and paths is not None:
            raise ValueError("Provide only one of `root` or `paths`, not both.")

        self.file_formats = self._normalize_and_validate_format(file_format)
        self.splits = list(splits) if splits is not None else None
        self.root = Path(root) if root is not None else None
        self.return_spkid = return_spkid

        rows = []

        if paths is not None:
            files = [Path(p) for p in paths if Path(p).is_file()]
            for p in files:
                rows.append(
                    AudioSample(
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
                        raise FileNotFoundError(
                            f"Split directory does not exist: {split_root}"
                        )
                    if not split_root.is_dir():
                        raise NotADirectoryError(f"Expected a directory: {split_root}")
                    search_roots.append((split, split_root))
            file_formats = (
                self.file_formats
                if self.file_formats is not None
                else self.VALID_FORMATS
            )
            for split, search_root in search_roots:
                for p in search_root.rglob("*"):
                    if not p.is_file():
                        continue
                    elif p.suffix.lower().lstrip(".") not in file_formats:
                        continue
                    rows.append(
                        AudioSample(
                            path=p,
                            split=split,
                            spk_id=self.get_spkid(p) if return_spkid else None,
                        )
                    )

        self.rows = sorted(rows, key=lambda row: str(row.path))

    @classmethod
    def _normalize_and_validate_format(
        cls, file_format: Optional[Union[str, Iterable[str]]]
    ) -> Optional[set[str]]:
        if file_format is None:
            return None

        if isinstance(file_format, str):
            formats = [file_format]
        else:
            formats = list(file_format)

        normalized = set()
        for fmt in formats:
            fmt = fmt.lower().strip()
            if fmt.startswith("."):
                fmt = fmt[1:]

            if fmt not in cls.VALID_FORMATS:
                valid = ", ".join(sorted(cls.VALID_FORMATS))
                raise ValueError(
                    f"Invalid audio format: {fmt!r}. Valid formats are: {valid}"
                )

            normalized.add(f".{fmt}")

        return normalized

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> AudioSample:
        return self.rows[idx]

    def get_spkid(self, file_path: Path):
        raise NotImplementedError(
            f"{type(self).__name__} must implement `get_spkid` when `return_spkid=True`."
        )

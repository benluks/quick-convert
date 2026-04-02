from pathlib import Path
from typing import Iterable, Optional, Union

from torch.utils.data import Dataset

from ..utils.audio import get_supported_formats


class BaseDataset(Dataset):
    """
    A PyTorch Dataset that returns audio file paths.

    Args:
        root: Directory to search recursively for audio files.
        paths: Optional explicit iterable of file paths. If provided, `root` is ignored.
        file_format: Optional extension or iterable of extensions, e.g. "wav", ".wav",
                     or ["wav", "flac"].

    Returns:
        str: The path to the audio file.
    """

    VALID_FORMATS = get_supported_formats()

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        paths: Optional[Iterable[Union[str, Path]]] = None,
        file_format: Optional[Union[str, Iterable[str]]] = None,
        load=False,
    ):
        if root is None and paths is None:
            raise ValueError("You must provide either `root` or `paths`.")
        if root is not None and paths is not None:
            raise ValueError("Provide only one of `root` or `paths`, not both.")

        self.file_formats = self._normalize_and_validate_format(file_format)

        if paths is not None:
            files = [Path(p) for p in paths]
            files = [p for p in files if p.is_file()]
        else:
            root = Path(root)
            if not root.exists():
                raise FileNotFoundError(f"Directory does not exist: {root}")
            if not root.is_dir():
                raise NotADirectoryError(f"Expected a directory: {root}")

            files = [p for p in root.rglob("*") if p.is_file()]

        self.root = root
        if self.file_formats is not None:
            files = [p for p in files if p.suffix.lower() in self.file_formats]
        else:
            files = [
                p
                for p in files
                if p.suffix and p.suffix.lower().lstrip(".") in self.VALID_FORMATS
            ]

        self.files = sorted(files)

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
        return len(self.files)

    def __getitem__(self, idx: int) -> str:
        return str(self.files[idx])

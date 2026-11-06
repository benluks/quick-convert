"""Publish complete files using temporary files on the destination filesystem."""

from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile


@contextmanager
def atomic_file(path: Path, mode: str = "w", *, replace: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(
            mode, encoding=None if "b" in mode else "utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            yield stream
        if replace:
            temporary.replace(path)
        else:
            path.hardlink_to(temporary)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_text(path: Path, text: str) -> None:
    with atomic_file(path) as stream:
        stream.write(text)

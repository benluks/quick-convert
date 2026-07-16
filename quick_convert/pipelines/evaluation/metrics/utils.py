from typing import Iterable, TextIO
import os


def load_lines(
    source: str | os.PathLike | TextIO | Iterable[str],
) -> list[str]:
    """
    Normalize a file path, file handle, or iterable of strings
    into a list of stripped lines.

    Examples:
        load_lines("refs.txt")
        load_lines(Path("hyps.txt"))
        load_lines(open("refs.txt"))
        load_lines(["hello", "world"])
    """

    if isinstance(source, (str, os.PathLike)):
        with open(source) as f:
            return [line.rstrip("\n") for line in f]

    if hasattr(source, "read"):
        return [line.rstrip("\n") for line in source]

    return [str(line).rstrip("\n") for line in source]

from __future__ import annotations

from functools import lru_cache
from os import PathLike
from typing import Set

import soundfile as sf

import torchaudio
import torchaudio.transforms as T


def load_audio(audio_path: PathLike, target_sr: int = None, mono: bool = False) -> tuple[float["1 t"], int]:
    x, sr = torchaudio.load(str(audio_path))
    if target_sr:
        x = T.Resample(sr, target_sr)(x)
    if mono and x.shape[-2] == 2:
        x = x.mean(dim=-2, keepdim=True)
    return x, sr


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def _is_audio_by_ext(path):
    return path.lower().endswith(tuple(AUDIO_EXTS))


def _is_audio_file(path):
    try:
        sf.info(path)
        return True
    except RuntimeError:
        return False


def is_audio(path):
    if not _is_audio_by_ext(path):
        return False
    return _is_audio_file(path)


class AudioBackendError(RuntimeError):
    """Raised when no usable audio backend can report supported formats."""


@lru_cache(maxsize=1)
def get_supported_formats() -> Set[str]:
    """
    Return a normalized set of supported audio file extensions.

    Strategy:
    1. Try torchaudio's sox extension.
    2. Fall back to soundfile/libsndfile.
    3. Raise a clear error if neither works.
    """
    formats: set[str] = set()
    errors: list[str] = []

    # 1) torchaudio + sox
    try:
        sox_ext = torchaudio._extension.lazy_import_sox_ext()
        sox_formats = sox_ext.list_read_formats()
        formats.update(fmt.lower().lstrip(".") for fmt in sox_formats)
    except Exception as e:
        errors.append(f"torchaudio sox backend unavailable: {type(e).__name__}: {e}")

    # 2) soundfile / libsndfile
    try:
        import soundfile as sf

        sf_formats = sf.available_formats().keys()
        formats.update(fmt.lower().lstrip(".") for fmt in sf_formats)
    except Exception as e:
        errors.append(f"soundfile backend unavailable: {type(e).__name__}: {e}")

    if formats:
        return formats

    raise AudioBackendError(
        "Could not determine supported audio formats because no usable audio "
        "backend was available.\n" + "\n".join(errors)
    )

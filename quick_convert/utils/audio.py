from __future__ import annotations

from functools import lru_cache
from os import PathLike

import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T

from quick_convert.types import AudioInput


def load_audio(
    audio_path: PathLike, target_sr: int | None = None, mono: bool = False, device="cpu"
) -> tuple[torch.Tensor, int]:
    x, sr = torchaudio.load(str(audio_path))
    if target_sr:
        x = T.Resample(sr, target_sr)(x)
    if mono and x.shape[-2] == 2:
        x = x.mean(dim=-2, keepdim=True)
    return x.to(device=device), sr


def load_audio_input(
    audio: AudioInput,
    *,
    target_sample_rate: int,
    sample_rate: int | None = None,
    mono: bool = True,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Load and normalize a path or in-memory waveform for inference.

    Tensor inputs are assumed to already use ``target_sample_rate`` when
    ``sample_rate`` is omitted. One-dimensional tensors are interpreted as
    mono waveforms and returned with an explicit channel dimension.
    """
    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate must be positive.")

    if isinstance(audio, torch.Tensor):
        waveform = audio
        source_sample_rate = target_sample_rate if sample_rate is None else sample_rate
    else:
        if sample_rate is not None:
            raise ValueError("sample_rate applies only to tensor inputs; file sample rates are read from the file.")
        waveform, source_sample_rate = load_audio(audio, mono=False, device="cpu")

    if source_sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim != 2:
        raise ValueError(f"Expected waveform shape (time,) or (channels, time), got {tuple(waveform.shape)}.")

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if source_sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_sample_rate, target_sample_rate)

    return waveform.to(device=device)


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
def get_supported_formats() -> set[str]:
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

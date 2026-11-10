from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from math import isfinite
from os import PathLike
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T

from quick_convert.types import AudioInput


def load_audio(
    audio_path: PathLike,
    target_sr: int | None = None,
    mono: bool = False,
    device="cpu",
    *,
    start: float | None = None,
    end: float | None = None,
) -> tuple[torch.Tensor, int]:
    """Decode audio, optionally reading a half-open interval in source seconds."""
    path = str(audio_path)
    if Path(path).suffix.lower() == ".mp4" and "ffmpeg" not in torchaudio.list_audio_backends():
        x, sr = _load_mp4_with_ffmpeg(path, start, end)
    elif start is None and end is None:
        x, sr = torchaudio.load(path)
    else:
        x, sr = _load_audio_segment(path, start, end)
    if target_sr and sr != target_sr:
        x = T.Resample(sr, target_sr)(x)
        sr = target_sr
    if mono and x.shape[-2] == 2:
        x = x.mean(dim=-2, keepdim=True)
    return x.to(device=device), sr


def _validate_segment(start: float | None, end: float | None) -> float:
    start = 0.0 if start is None else start
    if not isfinite(start) or start < 0:
        raise ValueError("Audio start must be finite and nonnegative.")
    if end is not None and (not isfinite(end) or end <= start):
        raise ValueError("Audio end must be finite and greater than start.")
    return start


def _segment_frames(start: float, end: float | None, sample_rate: int) -> tuple[int, int]:
    first = round(start * sample_rate)
    count = -1 if end is None else round(end * sample_rate) - first
    if end is not None and count <= 0:
        raise ValueError("Audio interval contains no source frames.")
    return first, count


def _check_audio_length(waveform: torch.Tensor, num_frames: int) -> None:
    if waveform.shape[-1] == 0 or (num_frames > 0 and waveform.shape[-1] != num_frames):
        raise ValueError("Audio interval exceeds the source recording.")


def _load_audio_segment(path: str, start: float | None, end: float | None) -> tuple[torch.Tensor, int]:
    start = _validate_segment(start, end)
    info = torchaudio.info(path)
    first_frame, num_frames = _segment_frames(start, end, info.sample_rate)
    if info.num_frames > 0 and (
        first_frame >= info.num_frames or (num_frames > 0 and first_frame + num_frames > info.num_frames)
    ):
        raise ValueError("Audio interval exceeds the source recording.")

    waveform, sample_rate = torchaudio.load(path, frame_offset=first_frame, num_frames=num_frames)
    _check_audio_length(waveform, num_frames)
    return waveform, sample_rate


def _run_audio_command(command: list[str]) -> bytes:
    try:
        return subprocess.run(command, capture_output=True, check=True).stdout
    except FileNotFoundError as error:
        raise AudioBackendError(
            "MP4 audio requires the TorchAudio FFmpeg backend or both ffmpeg and ffprobe on PATH."
        ) from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode(errors="replace").strip()
        raise AudioBackendError(f"{command[0]} could not read MP4 audio: {detail}") from error


def _load_mp4_with_ffmpeg(path: str, start: float | None, end: float | None) -> tuple[torch.Tensor, int]:
    start = _validate_segment(start, end)
    source = str(Path(path).absolute())
    metadata = _run_audio_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels",
            "-of",
            "json",
            source,
        ]
    )
    streams = json.loads(metadata).get("streams", [])
    if not streams:
        raise AudioBackendError(f"No audio stream in {path}.")
    try:
        sample_rate = int(streams[0]["sample_rate"])
        channels = int(streams[0]["channels"])
    except (KeyError, ValueError) as error:
        raise AudioBackendError(f"No usable audio stream metadata in {path}.") from error
    if sample_rate <= 0 or channels <= 0:
        raise AudioBackendError(f"No usable audio stream metadata in {path}.")
    first_frame, num_frames = _segment_frames(start, end, sample_rate)

    # Decode a short lead-in so predictive codecs can settle after a seek.
    seek_frame = max(0, first_frame - sample_rate)
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-ss",
        f"{seek_frame / sample_rate:.9f}",
        "-i",
        source,
        "-map",
        "0:a:0",
        "-ss",
        f"{(first_frame - seek_frame) / sample_rate:.9f}",
    ]
    if num_frames > 0:
        command.extend(["-t", f"{num_frames / sample_rate:.9f}"])
    command.extend(["-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"])
    samples = _run_audio_command(command)
    if not samples:
        raise ValueError("Audio interval exceeds the source recording.")
    waveform = torch.frombuffer(bytearray(samples), dtype=torch.float32).reshape(-1, channels).transpose(0, 1)
    _check_audio_length(waveform, num_frames)
    return waveform, sample_rate


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

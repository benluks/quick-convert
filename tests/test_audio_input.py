from pathlib import Path

import pytest
import torch

from quick_convert import AudioInput
from quick_convert.utils import audio as audio_utils


def test_audio_input_is_public() -> None:
    assert AudioInput is not None


def test_tensor_audio_defaults_to_target_sample_rate() -> None:
    waveform = torch.arange(8, dtype=torch.float32)

    loaded = audio_utils.load_audio_input(waveform, target_sample_rate=16_000)

    assert loaded.shape == (1, 8)
    assert torch.equal(loaded[0], waveform)


def test_tensor_audio_can_be_resampled(monkeypatch) -> None:
    calls = []

    def fake_resample(waveform, source_sample_rate, target_sample_rate):
        calls.append((source_sample_rate, target_sample_rate))
        return waveform[..., ::2]

    monkeypatch.setattr(audio_utils.torchaudio.functional, "resample", fake_resample)

    loaded = audio_utils.load_audio_input(
        torch.zeros(1, 8),
        sample_rate=32_000,
        target_sample_rate=16_000,
    )

    assert calls == [(32_000, 16_000)]
    assert loaded.shape == (1, 4)


def test_file_audio_uses_embedded_sample_rate(monkeypatch) -> None:
    monkeypatch.setattr(
        audio_utils,
        "load_audio",
        lambda path, **kwargs: (torch.zeros(2, 8), 48_000),
    )
    monkeypatch.setattr(
        audio_utils.torchaudio.functional,
        "resample",
        lambda waveform, source_sample_rate, target_sample_rate: waveform[..., ::3],
    )

    loaded = audio_utils.load_audio_input(Path("input.wav"), target_sample_rate=16_000)

    assert loaded.shape == (1, 3)


def test_file_audio_rejects_external_sample_rate() -> None:
    with pytest.raises(ValueError, match="applies only to tensor inputs"):
        audio_utils.load_audio_input(
            "input.wav",
            sample_rate=44_100,
            target_sample_rate=16_000,
        )


def test_audio_input_rejects_batched_tensor() -> None:
    with pytest.raises(ValueError, match="Expected waveform shape"):
        audio_utils.load_audio_input(torch.zeros(2, 1, 8), target_sample_rate=16_000)


def test_audio_input_rejects_invalid_tensor_sample_rate() -> None:
    with pytest.raises(ValueError, match="sample_rate must be positive"):
        audio_utils.load_audio_input(
            torch.zeros(8),
            sample_rate=0,
            target_sample_rate=16_000,
        )

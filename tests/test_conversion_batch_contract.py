from pathlib import Path

import pytest
import torch

from quick_convert.data import AudioBatch, AudioSample
from quick_convert.systems.conversion.base import generate_audio_batch


class LengthChangingConversion:
    sample_rate = sr = 16_000

    def convert(self, audio, *, sample_rate=None, **kwargs):
        if isinstance(audio, Path):
            length = int(audio.stem)
        else:
            length = audio.shape[-1] - 1
        return torch.ones(1, length)


def test_default_batch_adapter_preserves_generated_lengths_for_paths() -> None:
    batch = AudioBatch.from_samples(
        [
            AudioSample(utt_id="first", path=Path("7.wav")),
            AudioSample(utt_id="second", path=Path("4.wav")),
        ]
    )

    generated = generate_audio_batch(batch, LengthChangingConversion().convert, sample_rate=16000)

    assert generated.waveforms.shape == (2, 7)
    assert generated.lengths.tolist() == [7, 4]
    assert generated.waveform(1).shape == (4,)
    assert generated.sample_rate == 16_000


def test_default_batch_adapter_uses_unpadded_loaded_audio() -> None:
    batch = AudioBatch.from_samples(
        [
            AudioSample(
                utt_id="first",
                path=Path("first.wav"),
                waveform=torch.ones(1, 8),
                sample_rate=16_000,
            ),
            AudioSample(
                utt_id="second",
                path=Path("second.wav"),
                waveform=torch.ones(1, 5),
                sample_rate=16_000,
            ),
        ]
    )

    generated = generate_audio_batch(batch, LengthChangingConversion().convert, sample_rate=16000)

    assert generated.lengths.tolist() == [7, 4]


class StereoConversion(LengthChangingConversion):
    def convert(self, audio, *, sample_rate=None, **kwargs):
        return torch.zeros(2, 8)


def test_default_batch_adapter_rejects_multichannel_output() -> None:
    batch = AudioBatch.from_samples([AudioSample(utt_id="sample", path=Path("8.wav"))])

    with pytest.raises(ValueError, match="must generate mono waveforms"):
        generate_audio_batch(batch, StereoConversion().convert, sample_rate=16000)

from pathlib import Path

import pytest
import torch

from quick_convert.components.conditioning.reference_prompt import EnergyActivityDetector, SpeechPromptBuilder


def test_energy_removes_internal_silence_and_preserves_quiet_speech():
    detector = EnergyActivityDetector(threshold_db=-40, frame_seconds=0.01, padding_seconds=0)
    waveform = torch.cat((torch.zeros(20), torch.full((20,), 0.02), torch.zeros(20), torch.ones(15)))
    before = waveform.clone()
    assert detector(waveform, 1000) == ((20, 40), (60, 75))
    assert detector(torch.zeros(100), 1000) == ()
    torch.testing.assert_close(waveform, before)


def test_activity_padding_merges_intervals_and_clamps_to_audio():
    detector = EnergyActivityDetector(frame_seconds=0.01, padding_seconds=0.02)
    waveform = torch.cat((torch.ones(10), torch.zeros(20), torch.ones(10)))
    assert detector(waveform, 1000) == ((0, 40),)


def test_prompt_crossfades_joins_and_fills_duration_after_overlap():
    detector = EnergyActivityDetector(frame_seconds=0.01, padding_seconds=0)
    builder = SpeechPromptBuilder(detector, crossfade_seconds=0.01)
    first = torch.cat((torch.zeros(20), torch.ones(50), torch.zeros(20)))
    second = torch.cat((torch.zeros(10), -torch.ones(80)))
    state = torch.random.get_rng_state().clone()
    recordings = [(Path("a.wav"), first, 1000), (Path("b.wav"), second, 1000)]
    prompt = builder(recordings, 0.1)
    again = builder(recordings, 0.1)
    assert prompt.waveform.shape == (100,)
    assert [(s.start_sample, s.end_sample, s.overlap_samples) for s in prompt.segments] == [
        (20, 70, 0),
        (10, 70, 10),
    ]
    assert prompt.waveform[0] == prompt.waveform[-1] == 0
    # An abrupt +1/-1 join has a step of 2; the linear crossfade removes it.
    assert prompt.waveform.diff().abs().max() < 0.2
    assert prompt.waveform.abs().max() <= 1
    torch.testing.assert_close(prompt.waveform, again.waveform)
    torch.testing.assert_close(first[20:70], torch.ones(50))
    torch.testing.assert_close(second[10:], -torch.ones(80))
    torch.testing.assert_close(torch.random.get_rng_state(), state)


def test_prompt_detector_is_replaceable_and_stops_loading_when_full():
    calls = []

    def detector(waveform, sample_rate):
        calls.append(sample_rate)
        return ((5, 15),)

    def recordings():
        yield Path("a.wav"), torch.ones(20), 1000
        raise AssertionError("No more recordings should be loaded.")

    prompt = SpeechPromptBuilder(detector, crossfade_seconds=0)(recordings(), 0.01)
    assert prompt.waveform.numel() == 10
    assert calls == [1000]


def test_prompt_insufficient_audio_is_explicit():
    builder = SpeechPromptBuilder(EnergyActivityDetector(padding_seconds=0))
    with pytest.raises(ValueError, match="Not enough active reference audio"):
        builder([(Path("silent.wav"), torch.zeros(100), 1000)], 0.1)
    with pytest.raises(ValueError, match="retained 0.100s"):
        builder([(Path("short.wav"), torch.ones(100), 1000)], 0.2)


@pytest.mark.parametrize("intervals", [((-1, 4),), ((1, 21),), ((5, 10), (9, 12))])
def test_prompt_rejects_invalid_detector_intervals(intervals):
    builder = SpeechPromptBuilder(lambda waveform, rate: intervals)
    with pytest.raises(ValueError, match="sample intervals"):
        builder([(Path("bad.wav"), torch.ones(20), 1000)], 1)


def test_prompt_rejects_mixed_rates_and_invalid_parameters():
    with pytest.raises(ValueError, match="common sample rate"):
        SpeechPromptBuilder()([(Path("a"), torch.ones(20), 1000), (Path("b"), torch.ones(20), 2000)], 1)
    for value in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="crossfade_seconds"):
            SpeechPromptBuilder(crossfade_seconds=value)
    with pytest.raises(ValueError, match="threshold_db"):
        EnergyActivityDetector(threshold_db=0)

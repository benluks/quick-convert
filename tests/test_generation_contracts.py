import sys
from types import ModuleType

import pytest
import torch
from torch import nn

from quick_convert.components.decoders.cosyvoice import CosyVoiceSpectrogramGenerator
from quick_convert.data import GeneratedAudio


class FakeFlow(nn.Module):
    output_size = 80

    def inference(self, *, token, **kwargs):
        return torch.zeros(token.shape[0], self.output_size, token.shape[1] * 2), None

    def output_lengths(self, token_lengths):
        return token_lengths * 2


class FakeVocoder(nn.Module):
    samples_per_frame = 4
    sample_rate = 24_000

    def forward(self, mel):
        return torch.zeros(mel.shape[0], 1, mel.shape[-1] * self.samples_per_frame)


def make_decoder():
    decoder = CosyVoiceSpectrogramGenerator.__new__(CosyVoiceSpectrogramGenerator)
    nn.Module.__init__(decoder)
    decoder.device = torch.device("cpu")
    decoder.flow = FakeFlow()
    decoder.vocoder = FakeVocoder()
    decoder.input_projection = None
    return decoder


def test_generated_audio_returns_unpadded_waveforms():
    audio = GeneratedAudio(
        waveforms=torch.zeros(2, 12),
        lengths=torch.tensor([12, 7]),
        sample_rate=24_000,
    )

    assert audio.waveform(0).shape == (12,)
    assert audio.waveform(1).shape == (7,)


def test_generated_audio_rejects_lengths_beyond_padded_tensor():
    with pytest.raises(ValueError, match="exceeds the padded waveform size"):
        GeneratedAudio(
            waveforms=torch.zeros(1, 5),
            lengths=torch.tensor([6]),
            sample_rate=24_000,
        )


def test_cosyvoice_reports_exact_matcha_mel_lengths():
    sample_lengths = torch.tensor([320, 639, 640, 16_000])

    mel_lengths = CosyVoiceSpectrogramGenerator.mel_output_lengths(sample_lengths, sampling_rate=16_000)

    assert mel_lengths.tolist() == [1, 1, 2, 50]

    odd_padding_lengths = CosyVoiceSpectrogramGenerator.mel_output_lengths(
        torch.tensor([441, 442]), sampling_rate=22_050
    )
    assert odd_padding_lengths.tolist() == [0, 1]


def test_cosyvoice_mel_lengths_reject_invalid_sample_rate():
    with pytest.raises(ValueError, match="sampling_rate must be positive"):
        CosyVoiceSpectrogramGenerator.mel_output_lengths(torch.tensor([320]), sampling_rate=0)


def test_cosyvoice_generation_preserves_input_lengths_and_reports_outputs():
    decoder = make_decoder()
    input_lengths = torch.tensor([5, 3])

    output = decoder(
        feature=torch.zeros(2, 5, 4),
        length=input_lengths,
        speaker_embedding=torch.zeros(2, 2),
        run_vocoder=True,
    )

    assert input_lengths.tolist() == [5, 3]
    assert output.mel.shape == (2, 80, 10)
    assert output.mel_lengths.tolist() == [10, 6]
    assert output.audio is not None
    assert output.audio.waveforms.shape == (2, 40)
    assert output.audio.lengths.tolist() == [40, 24]
    assert output.audio.sample_rate == 24_000


def test_cosyvoice_construction_does_not_load_vocoder(monkeypatch):
    audio_module = ModuleType("quick_convert.external.matcha.utils.audio")
    audio_module.mel_spectrogram = object()
    monkeypatch.setitem(sys.modules, audio_module.__name__, audio_module)

    flow = FakeFlow()
    flow.vocab_size = 1
    flow.input_size = 4
    decoder = CosyVoiceSpectrogramGenerator(flow=flow, feature_dim=4, device="cpu")

    assert decoder.vocoder is None


def test_cosyvoice_default_vocoder_loads_once(monkeypatch):
    decoder = make_decoder()
    decoder.vocoder = None
    decoder.vocoder_repo_id = "repo"
    decoder.vocoder_filename = "hift.pt"
    calls = []

    class Loader:
        @classmethod
        def from_pretrained(cls, **kwargs):
            calls.append(kwargs)
            return FakeVocoder()

    module = ModuleType("quick_convert.components.decoders.hift_generator")
    module.CosyVoiceHiFTDecoder = Loader
    monkeypatch.setitem(sys.modules, module.__name__, module)

    first = decoder._get_vocoder()
    second = decoder._get_vocoder()

    assert first is second
    assert len(calls) == 1

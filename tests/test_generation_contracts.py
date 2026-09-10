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

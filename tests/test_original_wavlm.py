import pytest
import torch
from torch import nn

from quick_convert.components.encoders.content.microsoft_wavlm import MicrosoftWavLMContentEncoder


def test_original_wavlm_preserves_unpadded_waveform_and_frame_lengths(monkeypatch):
    class FakeWavLM(nn.Linear):
        def extract_features(self, waveform, *, output_layer):
            assert output_layer == 6
            return waveform[:, ::2, None].expand(-1, -1, 1024), None

    monkeypatch.setattr(MicrosoftWavLMContentEncoder, "_load_model", staticmethod(lambda: FakeWavLM(1, 1)))
    encoder = MicrosoftWavLMContentEncoder(device="cpu", do_normalize=False)
    waveform = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 999.0]])
    result = encoder.encode_waveforms(waveform, torch.tensor([5, 4]))

    assert result.lengths.tolist() == [3, 2]
    assert result.values[:, :, 0].tolist() == [[1.0, 3.0, 5.0], [6.0, 8.0, 0.0]]
    assert waveform[1, -1].item() == 999.0
    with pytest.raises(ValueError, match="16000 Hz"):
        encoder.encode_waveforms(waveform, sample_rate=24000)

import warnings

import pytest
import torch
from torch import nn


CosyVoiceHiFTDecoder = pytest.importorskip("quick_convert.components.decoders.hift_generator").CosyVoiceHiFTDecoder


class PitchGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.pitch = nn.Linear(1, 1)
        self.calls = []

    def inference(self, mel, finalize):
        self.calls.append(mel.device.type)
        self.pitch.double()
        return mel[:, 0], None


def make_vocoder():
    model = CosyVoiceHiFTDecoder.__new__(CosyVoiceHiFTDecoder)
    nn.Module.__init__(model)
    model.generator = PitchGenerator()
    return model


def test_cpu_vocoding_does_not_warn():
    model = make_vocoder()
    with warnings.catch_warnings(record=True) as caught:
        waveform = model(torch.ones(1, 80, 20))
    assert not caught
    assert waveform.device.type == "cpu"
    assert model.generator.calls == ["cpu"]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Requires Apple MPS")
def test_mps_vocoding_falls_back_once_and_survives_parent_device_move():
    model = make_vocoder().half().to("mps")
    mel = torch.ones(1, 80, 20, device="mps", dtype=torch.float16)
    with pytest.warns(RuntimeWarning, match="vocoding will run on CPU"):
        waveform = model(mel)
    assert waveform.device.type == "mps"
    torch.testing.assert_close(waveform, mel[:, 0].float())
    assert model.generator.pitch.weight.dtype == torch.float32
    model.to("mps")
    with warnings.catch_warnings(record=True) as caught:
        model(mel)
    assert not caught
    assert model.generator.calls == ["cpu", "cpu"]


def test_unrelated_vocoder_errors_propagate():
    model = make_vocoder()

    def fail(mel, finalize):
        raise RuntimeError("invalid mel")

    model.generator.inference = fail
    with pytest.raises(RuntimeError, match="invalid mel"):
        model(torch.ones(1, 80, 20))

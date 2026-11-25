import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from quick_convert.components.encoders.content import wavlm
from quick_convert.components.encoders.content._wavlm_checkpoint import load_microsoft_wavlm_large


def test_original_wavlm_preserves_unpadded_waveform_and_frame_lengths(monkeypatch):
    class FakeWavLM(nn.Module):
        config = SimpleNamespace(conv_stride=[2], conv_kernel=[1])

        def forward(self, *, input_values, output_hidden_states, return_dict):
            assert len(input_values) == 1 and output_hidden_states and return_dict
            features = input_values[:, ::2, None].expand(-1, -1, 1024)
            return SimpleNamespace(hidden_states=(None,) * 6 + (features,))

    class Inputs(dict):
        def to(self, device):
            return self

    def processor(waveforms, *, return_attention_mask, **kwargs):
        assert not return_attention_mask
        return Inputs(input_values=torch.tensor(waveforms[0])[None])

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoFeatureExtractor=SimpleNamespace(from_pretrained=lambda *args, **kwargs: processor),
            AutoConfig=SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeWavLM.config),
            AutoModel=SimpleNamespace(from_config=lambda config: FakeWavLM()),
        ),
    )
    monkeypatch.setattr(wavlm, "load_microsoft_wavlm_large", lambda *args, **kwargs: None)
    encoder = wavlm.WavLMContentEncoder(use_original_checkpoint=True, layer=6, device="cpu")
    waveform = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 999.0]])
    result = encoder.encode_waveforms(waveform, torch.tensor([5, 4]))

    assert result.lengths.tolist() == [3, 2]
    assert result.values[:, :, 0].tolist() == [[1.0, 3.0, 5.0], [6.0, 8.0, 0.0]]
    assert waveform[1, -1].item() == 999.0
    with pytest.raises(ValueError, match="16000 Hz"):
        encoder.encode_waveforms(waveform, sample_rate=24000)


def test_original_wavlm_checkpoint_maps_and_requires_encoder_weights(monkeypatch):
    model = nn.Module()
    model.feature_projection = nn.Module()
    model.feature_projection.projection = nn.Linear(2, 1)
    checkpoint = {
        "model": {
            "post_extract_proj.weight": torch.ones(1, 2),
            "post_extract_proj.bias": torch.zeros(1),
            "final_proj.weight": torch.ones(1, 2),
        }
    }
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", lambda *args, **kwargs: checkpoint)
    load_microsoft_wavlm_large(model, local_files_only=False)
    torch.testing.assert_close(model.feature_projection.projection.weight, torch.ones(1, 2))

    del checkpoint["model"]["post_extract_proj.bias"]
    with pytest.raises(RuntimeError, match="Missing key"):
        load_microsoft_wavlm_large(model, local_files_only=False)

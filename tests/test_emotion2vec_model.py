import os

import pytest
import torch

from quick_convert.components.layers import LayerWeightedSum
from quick_convert.components.ssl import EmotionEncoder


@pytest.mark.skipif(os.environ.get("QC_ACOUSTIC_MODEL_TESTS") != "1", reason="Optional checkpoint download")
def test_pretrained_emotion2vec_layers_match_final_and_train_fusion():
    torch.set_num_threads(2)
    encoder = EmotionEncoder(layer=None, device="cpu")
    audio = torch.randn(2, 16000) * 0.01
    lengths = torch.tensor([16000, 12000])
    layered = encoder.encode_waveforms(audio, lengths)
    assert layered.values.ndim == 4
    assert layered.values.shape[2] == encoder.N_LAYERS
    assert layered.feature_dim == encoder.feature_dim
    assert layered.lengths[0] > layered.lengths[1] > 0
    assert torch.isfinite(layered.values).all()
    encoder.layer = -1
    final = encoder.encode_waveforms(audio, lengths)
    torch.testing.assert_close(layered.values[:, :, -1], final.values, rtol=1e-4, atol=1e-5)
    assert torch.equal(layered.lengths, final.lengths)
    fusion = LayerWeightedSum(num_layers=encoder.N_LAYERS)
    fusion(layered.values.detach().clone()).square().mean().backward()
    assert fusion.weights.grad is not None
    assert torch.isfinite(fusion.weights.grad).all()

from pathlib import Path

import torch
from torch import nn

from quick_convert.components.layers.heads import HeadOutput
from quick_convert.components.layers.rvq import RVQLosses, RVQOutput
from quick_convert.components.mixins.resource import ResolvedResource
from quick_convert.data import AudioBatch
from quick_convert.systems.asr import VQASRResult, VQASRSystem


class FakeQuantizer(nn.Module):
    n_codebooks = 1
    codebook_size = 8

    @staticmethod
    def output_lengths(lengths):
        return lengths

    def forward(self, features, padding_mask):
        assert features.shape == (2, 4, 3)
        assert padding_mask.tolist() == [[True, True, True], [True, True, False]]
        zero = features.new_zeros(())
        return RVQOutput(
            z_q=features + 1,
            layer_z_qs=[features + 1],
            codes=torch.zeros(2, 1, 3, dtype=torch.long),
            latents=features - 1,
            loss=RVQLosses(loss=zero),
        )


class FakeCTCHead(nn.Module):
    def predict(self, features, *, lengths, padding_mask):
        assert lengths.tolist() == [3, 2]
        assert padding_mask.tolist() == [[True, True, True], [True, True, False]]
        logits = torch.cat([features, features[..., :1]], dim=-1)
        return HeadOutput(predictions=logits, states={"logits": logits})


def make_batch(content):
    return AudioBatch(
        utt_ids=["a", "b"],
        paths=[Path("a.wav"), Path("b.wav")],
        splits=[None, None],
        resources={"content": content},
    )


def test_vq_asr_system_returns_predictions_and_probeable_representations():
    system = VQASRSystem(
        quantizer=FakeQuantizer(),
        ctc_head=FakeCTCHead(),
    )
    content = ResolvedResource(
        values=torch.ones(2, 3, 4),
        lengths=torch.tensor([3, 2]),
    )

    result = system(make_batch(content))

    assert isinstance(result, VQASRResult)
    assert result.logits.shape == (2, 3, 5)
    assert result.lengths.tolist() == [3, 2]
    assert result.contextual.shape == (2, 3, 4)
    assert result.z_q.shape == (2, 4, 3)
    assert result.latents.shape == (2, 4, 3)
    assert result.codes.shape == (2, 1, 3)
    torch.testing.assert_close(result.contextual, result.latents.transpose(1, 2))


def test_vq_asr_system_accepts_raw_feature_tensors():
    system = VQASRSystem(
        quantizer=FakeQuantizer(),
        ctc_head=FakeCTCHead(),
        use_latents=False,
    )

    result = system.encode_features(
        torch.ones(2, 3, 4),
        torch.tensor([3, 2]),
    )

    torch.testing.assert_close(result.contextual, result.z_q.transpose(1, 2))


def test_vq_asr_system_rejects_multiple_codebooks():
    quantizer = FakeQuantizer()
    quantizer.n_codebooks = 2

    try:
        VQASRSystem(quantizer=quantizer, ctc_head=FakeCTCHead())
    except ValueError as error:
        assert "single active codebook" in str(error)
    else:
        raise AssertionError("Expected a multi-codebook VQ-ASR system to be rejected.")

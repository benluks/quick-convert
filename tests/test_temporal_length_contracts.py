from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from quick_convert.components.encoders import ConformerEncoder, ConformerEncoderSSL, ParallelConformerEncoder
from quick_convert.components.layers import LayerWeightedSum
from quick_convert.components.layers.heads import HeadOutput
from quick_convert.components.layers.rvq_ema import ResidualVectorQuantizerEMA
from quick_convert.components.mixins.resource import ResolvedResource
from quick_convert.data import AudioBatch
from quick_convert.systems.asr import VQASRSystem
from quick_convert.systems.conversion import SSLReconstructionConversionSystem
from quick_convert.utils.masking import make_padding_mask


class CaptureHead(nn.Module):
    def predict(self, features, **kwargs):
        return HeadOutput(predictions=features)


@pytest.mark.parametrize(
    "module_type",
    [
        LayerWeightedSum,
        ConformerEncoder,
        ConformerEncoderSSL,
        ParallelConformerEncoder,
        ResidualVectorQuantizerEMA,
    ],
)
def test_time_preserving_modules_report_unchanged_lengths(module_type):
    lengths = torch.tensor([12, 7])

    assert torch.equal(module_type.output_lengths(lengths), lengths)


def test_layer_weighted_sum_preserves_time_and_projects_features():
    fusion = LayerWeightedSum(num_layers=3, projection=nn.Linear(4, 2))

    output = fusion(torch.zeros(2, 5, 3, 4))

    assert output.shape == (2, 5, 2)


def test_layer_weighted_sum_rejects_wrong_layer_count():
    fusion = LayerWeightedSum(num_layers=3)

    with pytest.raises(ValueError, match="Expected 3 representation layers"):
        fusion(torch.zeros(2, 5, 2, 4))


@pytest.mark.parametrize("task", ["reconstruction", "asr"])
@pytest.mark.parametrize("source", ["layered", "single", "rvq"])
def test_adaptation_matches_explicit_composition_and_keeps_gradients(task, source):
    torch.manual_seed(115)
    layered = source != "single"
    values = torch.randn((2, 5, 3, 4) if layered else (2, 5, 4), requires_grad=True)
    lengths = torch.tensor([4, 2])
    transform = LayerWeightedSum(3, projection=nn.Linear(4, 6)) if layered else nn.Identity()
    dim = 6 if layered else 4
    quantizer = (
        ResidualVectorQuantizerEMA(input_dim=dim, n_codebooks=1, codebook_size=4, codebook_dim=2, kmeans_init=False)
        if task == "asr" or source == "rvq"
        else None
    )
    if task == "asr":
        system = VQASRSystem(quantizer, CaptureHead(), layer_fusion=transform)
    else:
        system = SSLReconstructionConversionSystem(nn.Identity(), feature_transform=transform, encoder=quantizer)
    with torch.no_grad():
        expected = transform(values)
        if task == "asr":
            expected = expected[:, : int(lengths.max())]
        if quantizer is not None:
            manual = deepcopy(quantizer)(
                expected.transpose(1, 2), make_padding_mask(lengths, max_length=expected.shape[1])
            )
            expected = (manual.latents if task == "asr" else manual.z_q).transpose(1, 2)
    if task == "asr":
        result = system.encode_features(values, lengths, metadata={"feature_dim": 4, "frame_hz": 50 if layered else 25})
        actual, actual_lengths = result.contextual, result.lengths
    else:
        actual, actual_lengths, _ = system.encode_features(
            values, lengths, metadata={"feature_dim": 4, "frame_hz": 50 if layered else 25}
        )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_lengths, lengths)
    actual.square().mean().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all() and values.grad.abs().sum() > 0
    assert lengths.tolist() == [4, 2]


@pytest.mark.parametrize("task", ["reconstruction", "asr"])
@pytest.mark.parametrize("stage", ["transform", "quantizer"])
def test_invalid_stage_lengths_fail_before_the_next_component(task, stage):
    class BadTransform(nn.Identity):
        def forward(self, x):
            return x[:, :1] if stage == "transform" else x

    class BadQuantizer(ResidualVectorQuantizerEMA):
        @staticmethod
        def output_lengths(lengths):
            return lengths.float()

    quantizer = BadQuantizer(input_dim=4, n_codebooks=1, codebook_size=4, codebook_dim=2, kmeans_init=False).eval()
    if task == "asr":
        system = VQASRSystem(quantizer, CaptureHead(), layer_fusion=BadTransform())
    else:
        system = SSLReconstructionConversionSystem(nn.Identity(), feature_transform=BadTransform(), encoder=quantizer)
    with pytest.raises((ValueError, TypeError), match="padded time|integer dtype"):
        system.encode_features(torch.ones(2, 3, 4), torch.tensor([3, 2]))


@pytest.mark.parametrize("task", ["reconstruction", "asr"])
@pytest.mark.parametrize(
    "metadata", [{"feature_dim": 8}, {"temporal_granularity": "utterance"}, {"representation_type": "discrete"}]
)
def test_source_metadata_is_checked_before_adaptation(task, metadata):
    if task == "asr":
        system = VQASRSystem(nn.Identity(), CaptureHead())
    else:
        system = SSLReconstructionConversionSystem(nn.Identity())
    batch = AudioBatch(
        utt_ids=["a", "b"],
        paths=[Path("a"), Path("b")],
        splits=[None, None],
        resources={
            "content": ResolvedResource(torch.ones(2, 3, 4), torch.tensor([3, 2]), metadata),
            "speaker": torch.ones(2, 1),
        },
    )
    with pytest.raises(ValueError, match="metadata|frame-level|token IDs"):
        system(batch)


def test_frame_rate_checks_require_explicit_preservation_and_do_not_change_source_metadata():
    decoder = nn.Identity()
    decoder.flow = SimpleNamespace(input_frame_rate=50)
    metadata = {"frame_hz": 25.0, "feature_dim": 4}
    system = SSLReconstructionConversionSystem(decoder)
    values, lengths = torch.ones(2, 3, 4), torch.tensor([3, 2])
    with pytest.raises(ValueError, match="frame rate"):
        system.encode_features(values, lengths, metadata=metadata)

    # A custom transform's output lengths do not establish its output clock.
    class UnknownClock(nn.Module):
        def forward(self, x):
            return x

        @staticmethod
        def output_lengths(lengths):
            return lengths

    system.feature_transform = UnknownClock()
    system.encode_features(values, lengths, metadata=metadata)
    assert metadata == {"frame_hz": 25.0, "feature_dim": 4}

"""Pre-pooling ONNX output and waveform-prefix extraction without future context."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from quick_convert.components.encoders.speaker import CosyVoiceCAMPPlusSpeakerEncoder
from quick_convert.components.generators.acoustic import ConditioningStream, StreamConditioning
from quick_convert.components.resources.resolution import ResolvedResource
from quick_convert.data import AudioBatch
from quick_convert.systems.conversion import SSLReconstructionConversionSystem


@pytest.fixture
def frame_encoder(tmp_path, monkeypatch):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    helper = onnx.helper
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["input"], ["mean"], axes=[2], keepdims=0),
            helper.make_node("Unsqueeze", ["mean"], ["single"], axes=[1]),
            helper.make_node("Tile", ["single", "repeats"], ["frames"], name="frame_output"),
            helper.make_node(
                "ReduceMean", ["frames"], ["output"], axes=[2], keepdims=0, name="/xvector/stats/ReduceMean"
            ),
        ],
        "campplus_fixture",
        [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, "time", 80])],
        [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 512])],
        initializer=[helper.make_tensor("repeats", onnx.TensorProto.INT64, [3], [1, 512, 1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)], ir_version=8)
    path = tmp_path / "campplus.onnx"
    onnx.save(model, path)
    original = path.read_bytes()
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda **kw: str(path))
    encoder = CosyVoiceCAMPPlusSpeakerEncoder(device="cpu", output="frames")
    assert path.read_bytes() == original
    assert encoder.session.get_outputs()[0].name == "frames"
    assert encoder.feature_dim == 512 and encoder.frame_hz == 50 and encoder.sample_rate == 16000
    return encoder


def audio_batch(waveforms, lengths=None):
    count = waveforms.shape[0]
    return AudioBatch(
        utt_ids=[str(i) for i in range(count)],
        paths=[Path("audio")] * count,
        splits=[None] * count,
        resources={},
        waveforms=waveforms,
        lengths=lengths if lengths is not None else torch.full((count,), waveforms.shape[1]),
        sample_rates=torch.full((count,), 16000),
    )


def test_crop_happens_before_encoder_and_zero_prefix_skips_inference(frame_encoder, monkeypatch):
    calls = []

    def encode(waveform, frames):
        calls.append(waveform.clone())
        return torch.ones(frames, 512)

    monkeypatch.setattr(frame_encoder, "_encode_frames", encode)
    batch = audio_batch(torch.arange(1600.0).repeat(2, 1))
    original = batch.waveforms.clone()
    result = frame_encoder.encode_prefixes(batch, torch.tensor([3, 0]))
    assert result.values.shape == (2, 3, 512)
    assert result.lengths.tolist() == [3, 0]
    assert len(calls) == 1 and calls[0].numel() == 960
    assert result.values[1].count_nonzero() == 0
    torch.testing.assert_close(batch.waveforms, original)
    assert frame_encoder.encode_prefixes(batch, torch.tensor([0, 0])).values.shape == (2, 0, 512)


def test_prefix_encoding_is_invariant_to_future_audio_and_batch_padding(frame_encoder):
    torch.manual_seed(7)
    waveforms = torch.randn(2, 3200)
    waveforms[1, :1600] = waveforms[0, :1600]
    original = waveforms.clone()
    a = frame_encoder.encode_prefixes(audio_batch(waveforms), torch.tensor([5, 5]))
    torch.testing.assert_close(a.values[0], a.values[1])
    alone = frame_encoder(audio_batch(waveforms[:1, :1600]))
    torch.testing.assert_close(a.values[:1], alone.values)
    assert a.values.shape == (2, 5, 512)
    assert torch.isfinite(a.values).all()
    torch.testing.assert_close(waveforms, original)
    short = frame_encoder.encode_prefixes(audio_batch(waveforms[:1]), torch.tensor([1]))
    assert short.values.shape == (1, 1, 512)


def test_invalid_prefix_inputs_are_rejected(frame_encoder):
    batch = audio_batch(torch.ones(1, 1600))
    with pytest.raises(ValueError, match="exceeds"):
        frame_encoder.encode_prefixes(batch, torch.tensor([6]))
    with pytest.raises(ValueError, match="int64"):
        frame_encoder.encode_prefixes(batch, torch.tensor([1.0]))
    batch.sample_rates = torch.tensor([8000])
    with pytest.raises(ValueError, match="16 kHz"):
        frame_encoder.encode_prefixes(batch, torch.tensor([1]))


def test_system_samples_prefix_before_contextual_encoder_and_does_not_resample_it():
    conditioning = StreamConditioning(
        {"frame_speaker": 2}, prefix_fraction=(0.5, 0.5), context_mask_fraction=(0, 0), stream_drop_probability=0
    )

    class Decoder(nn.Module):
        stream_conditioning = conditioning
        mel_config = SimpleNamespace(sample_rate=16000, hop_size=320)
        flow = SimpleNamespace(token_mel_ratio=1)

        def mel_output_lengths(self, lengths, rate):
            return lengths // 320

    class ContextualEncoder(nn.Module):
        def encode_prefixes(self, batch, frame_lengths, frame_rate):
            assert frame_lengths.tolist() == [5, 3]
            assert frame_rate == 50
            return ResolvedResource(torch.ones(2, 5, 2), frame_lengths)

        def forward(self, batch):
            raise AssertionError("must not encode complete training waveforms")

    system = SSLReconstructionConversionSystem(Decoder(), online_encoders={"frame_speaker": ContextualEncoder()})
    batch = audio_batch(torch.ones(2, 3200), torch.tensor([3200, 1920]))
    streams = system.conditioning_streams(batch, torch.tensor([10, 6]))
    assert streams["frame_speaker"].is_prefix
    packed, score = conditioning.training_conditions(streams, torch.tensor([10, 6]), torch.zeros(2, 10, 2))
    assert score.sum(1).tolist() == [5, 3]
    assert packed[0, 5:].count_nonzero() == 0
    assert packed[1, 3:].count_nonzero() == 0


def test_explicit_prefix_does_not_get_shortened_again():
    conditioning = StreamConditioning(
        {"speaker": 1}, prefix_fraction=(0, 0), stream_drop_probability=1, context_mask_fraction=(0, 0)
    )
    streams = {"speaker": ConditioningStream(torch.ones(1, 3, 1), torch.tensor([3]), is_prefix=True)}
    packed, score = conditioning.training_conditions(streams, torch.tensor([8]), torch.zeros(1, 8, 1))
    assert score.sum() == 5 and packed[0, :3, 1].sum() == 3
    with pytest.raises(ValueError, match="continuation"):
        conditioning.training_conditions(streams, torch.tensor([3]), torch.zeros(1, 3, 1))


@pytest.mark.parametrize("backbone", ["cmdiff", "dit"])
def test_campp_run_config_includes_frame_speaker_and_no_xvector(backbone):
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from quick_convert.utils.resolvers import register_config_resolvers

    register_config_resolvers()
    with initialize_config_dir(
        version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / "quick_convert/configs")
    ):
        cfg = compose(config_name=f"run/train_sslr_w2vbert_{backbone}_campp_prefix")
        conditioning = instantiate(cfg.system.decoder.stream_conditioning)
        assert conditioning.dimensions == {"frame_speaker": 512}
        assert cfg.system.online_encoders.speaker is None
        assert cfg.system.decoder.flow.decoder.estimator.spk_dim is None
        assert cfg.system.decoder.flow.decoder.estimator.cond_dim == 513
        if backbone == "cmdiff":
            assert cfg.system.decoder.flow.decoder.estimator.in_channels == 673

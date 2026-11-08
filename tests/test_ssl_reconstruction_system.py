from pathlib import Path

import pytest
import torch
from torch import nn

from quick_convert.components.encoders.content.base import ContentFeatures
from quick_convert.components.generators.acoustic import CosyVoiceGenerationOutput
from quick_convert.components.layers.rvq import RVQLosses, RVQOutput
from quick_convert.components.resources.resolution import ResolvedResource
from quick_convert.data import AudioBatch, GeneratedAudio
from quick_convert.data.resources.base import TensorResourceBatch
from quick_convert.systems.conversion import SSLReconstructionConversionSystem, SSLReconstructionResult


class FakeFeatureTransform(nn.Module):
    @staticmethod
    def output_lengths(lengths):
        return lengths

    def forward(self, values):
        return values + 1


class FakeEncoder(nn.Module):
    @staticmethod
    def output_lengths(lengths):
        return lengths

    def forward(self, features, padding_mask):
        z_q = features + 2
        zero = features.new_zeros(())
        return RVQOutput(
            z_q=z_q,
            layer_z_qs=[z_q],
            codes=torch.zeros(features.shape[0], 1, features.shape[2], dtype=torch.long),
            latents=features,
            loss=RVQLosses(loss=zero),
        )


class FakeDecoder(nn.Module):
    def forward(self, *, feature, length, speaker_embedding, run_vocoder):
        assert feature.shape[0] == length.shape[0] == speaker_embedding.shape[0]
        assert feature.shape[-1] == 4
        assert speaker_embedding.shape[-1] == 5
        audio = None
        if run_vocoder:
            audio = GeneratedAudio(
                waveforms=torch.ones(feature.shape[0], int(length.max()) * 2),
                lengths=length * 2,
                sample_rate=24_000,
            )
        return CosyVoiceGenerationOutput(
            mel=torch.ones(feature.shape[0], 80, int(length.max())),
            mel_lengths=length,
            audio=audio,
        )


class FakeContentEncoder(nn.Module):
    sample_rate = 16_000
    device = torch.device("cpu")

    def forward(self, batch):
        return ResolvedResource(
            values=torch.ones(len(batch), 3, 4),
            lengths=torch.full((len(batch),), 3),
        )


class TrainableContentEncoder(nn.Module):
    sample_rate = 16_000
    device = torch.device("cpu")

    def __init__(self, wrapped=False):
        super().__init__()
        self.wrapped = wrapped
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        values = self.scale * torch.ones(len(batch), 3, 4)
        lengths = torch.full((len(batch),), 3)
        if self.wrapped:
            return ContentFeatures(
                values=values,
                lengths=lengths,
                feature_dim=4,
                representation_type="continuous",
                temporal_granularity="frame",
                backend="test",
                model_name="tiny",
                layer=None,
            )
        return ResolvedResource(values=values, lengths=lengths)


class FakeSpeakerEncoder(nn.Module):
    sample_rate = 16_000

    def forward(self, batch):
        return torch.ones(len(batch), 5)


def make_system():
    return SSLReconstructionConversionSystem(
        decoder=FakeDecoder(),
        feature_transform=FakeFeatureTransform(),
        encoder=FakeEncoder(),
    )


def test_ssl_reconstruction_generates_from_plain_feature_tensors():
    result = make_system().generate_features(
        torch.ones(2, 3, 4),
        torch.tensor([3, 2]),
        torch.ones(2, 5),
    )

    assert isinstance(result, SSLReconstructionResult)
    assert result.encoder_output is not None
    torch.testing.assert_close(result.features, torch.full((2, 3, 4), 4.0))
    assert result.generation.audio is not None
    assert result.generation.audio.lengths.tolist() == [6, 4]


def test_ssl_reconstruction_resolves_resources_from_audio_batches():
    batch = AudioBatch(
        utt_ids=["a", "b"],
        paths=[Path("a.wav"), Path("b.wav")],
        splits=[None, None],
        resources={
            "content": ResolvedResource(torch.ones(2, 3, 4), torch.tensor([3, 2])),
            "speaker": ResolvedResource(torch.ones(2, 5)),
        },
    )

    result = make_system()(batch, run_vocoder=False)

    assert result.generation.audio is None
    assert result.lengths.tolist() == [3, 2]


def test_ssl_reconstruction_accepts_a_plain_waveform_tensor():
    system = SSLReconstructionConversionSystem(
        decoder=FakeDecoder(),
        feature_transform=FakeFeatureTransform(),
        online_encoders={
            "content": FakeContentEncoder(),
            "speaker": FakeSpeakerEncoder(),
        },
    )

    result = system.reconstruct(
        torch.ones(160),
        sample_rate=16_000,
        run_vocoder=False,
    )

    assert result.generation.audio is None
    assert result.features.shape == (1, 3, 4)
    assert result.lengths.tolist() == [3]


def make_audio_batch():
    return AudioBatch(
        utt_ids=["a"],
        paths=[Path("a.wav")],
        splits=[None],
        resources={},
        waveforms=torch.ones(1, 160),
        lengths=torch.tensor([160]),
        sample_rates=torch.tensor([16_000]),
    )


@pytest.mark.parametrize("wrapped", [False, True])
def test_online_encoder_is_frozen_and_detached_by_default(wrapped):
    encoder = TrainableContentEncoder(wrapped=wrapped)
    system = SSLReconstructionConversionSystem(
        decoder=FakeDecoder(),
        online_encoders={"content": encoder},
    )

    resource = system.get_resource(make_audio_batch(), "content")

    assert not encoder.scale.requires_grad
    assert not resource.values.requires_grad
    assert (resource.metadata is not None) is wrapped


@pytest.mark.parametrize("wrapped", [False, True])
def test_trainable_online_encoder_preserves_gradients_and_updates(wrapped):
    encoder = TrainableContentEncoder(wrapped=wrapped)
    system = SSLReconstructionConversionSystem(
        decoder=FakeDecoder(),
        online_encoders={"content": encoder},
        trainable_online_encoders=("content",),
    )
    optimizer = torch.optim.SGD(system.parameters(), lr=0.1)
    before = encoder.scale.detach().clone()

    resource = system.get_resource(make_audio_batch(), "content")
    loss = resource.values.square().mean()
    loss.backward()

    assert encoder.scale.requires_grad
    assert encoder.scale.grad is not None
    assert encoder.scale.grad.norm() > 0
    if wrapped:
        assert resource.metadata["model_name"] == "tiny"
        assert resource.metadata["frame_hz"] is None
    else:
        assert resource.metadata is None

    optimizer.step()

    assert not torch.equal(before, encoder.scale.detach())


def test_precomputed_resources_remain_detached_with_trainable_online_encoder():
    encoder = TrainableContentEncoder()
    system = SSLReconstructionConversionSystem(
        decoder=FakeDecoder(),
        online_encoders={"content": encoder},
        trainable_online_encoders=("content",),
    )
    batch = make_audio_batch()
    source = torch.ones(1, 3, 4, requires_grad=True)
    batch.resources["content"] = ResolvedResource(source, torch.tensor([3]))

    resource = system.get_resource(batch, "content")

    assert not resource.values.requires_grad


def test_precomputed_content_matches_online_values_and_takes_precedence(monkeypatch):
    encoder = TrainableContentEncoder(wrapped=True)
    system = SSLReconstructionConversionSystem(decoder=FakeDecoder(), online_encoders={"content": encoder})
    batch = make_audio_batch()
    online = system.get_resource(batch, "content")
    batch.resources["content"] = TensorResourceBatch(online.values, online.lengths)

    def unexpected_encode(batch):
        raise AssertionError("Loaded content must take precedence over the online encoder.")

    monkeypatch.setattr(encoder, "forward", unexpected_encode)
    precomputed = system.get_resource(batch, "content")

    torch.testing.assert_close(precomputed.values, online.values)
    torch.testing.assert_close(precomputed.lengths, online.lengths)
    assert online.metadata["model_name"] == "tiny"
    assert precomputed.metadata is None


@pytest.mark.parametrize("backend_inference", [False, True])
def test_frozen_content_can_feed_trainable_downstream_layers(backend_inference):
    encoder = TrainableContentEncoder()
    if backend_inference:
        encoder.forward = torch.inference_mode()(encoder.forward)
    system = SSLReconstructionConversionSystem(decoder=FakeDecoder(), online_encoders={"content": encoder})
    values = system.get_resource(make_audio_batch(), "content").values
    projection = nn.Linear(4, 2)
    projection(values).square().mean().backward()
    assert projection.weight.grad is not None
    assert not torch.is_inference(values)
    assert encoder.scale.grad is None


@pytest.mark.parametrize("trainable", [False, True])
def test_training_mode_preserves_frozen_encoders_and_vocoder(trainable):
    encoder = TrainableContentEncoder()
    encoder.dropout = nn.Dropout(0.5)
    decoder = FakeDecoder()
    decoder.vocoder = nn.Linear(4, 4).requires_grad_(False)
    system = SSLReconstructionConversionSystem(
        decoder=decoder,
        online_encoders={"content": encoder},
        trainable_online_encoders=("content",) if trainable else (),
    )
    assert system.train() is system
    assert encoder.training is trainable
    assert encoder.dropout.training is trainable
    assert not decoder.vocoder.training
    assert decoder.training
    system.eval()
    assert not encoder.training


def test_native_reference_conversion_prepares_content_mels_and_global_embedding(tmp_path, monkeypatch):
    from quick_convert.components.conditioning.reference_prompt import SpeechPromptBuilder
    from quick_convert.components.generators.acoustic.cosyvoice_reference import CosyVoiceReference
    from quick_convert.components.transforms.speaker_aggregation import SpeakerEmbeddingAggregation
    from quick_convert.systems.conversion.reference import ReferenceCondition
    from quick_convert.systems.conversion.targets import ReferenceSpeech

    class NativeDecoder(FakeDecoder):
        stream_conditioning = None
        requires_speaker_embedding = True

        def prepare_reference(self, features, lengths, waveform, waveform_lengths, sample_rate):
            assert waveform_lengths.tolist() == [1600]
            return CosyVoiceReference(features.clone(), lengths.clone(), torch.ones(1, 3, 80), lengths.clone())

        def forward(self, *, feature, length, speaker_embedding, run_vocoder, reference):
            assert reference.features.shape == (2, 3, 4)
            assert reference.mel_lengths.tolist() == [3, 3]
            torch.testing.assert_close(speaker_embedding, torch.full((2, 5), 7 / 3))
            reference.features.zero_()
            speaker_embedding.zero_()
            return super().forward(
                feature=feature, length=length, speaker_embedding=speaker_embedding, run_vocoder=run_vocoder
            )

    class Speaker(nn.Module):
        sample_rate = 16000

        def forward(self, batch):
            return batch.waveforms[:, : int(batch.lengths[0])].mean(1, keepdim=True).expand(-1, 5)

    def load(path, **kwargs):
        return torch.full((1, 3200 if Path(path).stem == "b" else 1600), 3.0 if Path(path).stem == "b" else 1.0)

    monkeypatch.setattr("quick_convert.systems.conversion.ssl_reconstruction.load_audio_input", load)
    system = SSLReconstructionConversionSystem(
        NativeDecoder(),
        online_encoders={
            "content": FakeContentEncoder(),
            "speaker": Speaker(),
        },
    ).eval()
    request = ReferenceSpeech(
        [tmp_path / "b.wav", tmp_path / "a.wav"],
        prompt_seconds=0.1,
        prompt_builder=SpeechPromptBuilder(),
        aggregation=SpeakerEmbeddingAggregation("duration_weighted"),
    )
    target = system.prepare_target(request)
    assert target.durations == (0.1, 0.2)
    assert target.prompt_segments
    batch = AudioBatch(
        utt_ids=["x", "y"],
        paths=[Path("x"), Path("y")],
        splits=[None, None],
        resources={},
        waveforms=torch.zeros(2, 1600),
        lengths=torch.tensor([1600, 1600]),
        sample_rates=torch.tensor([16000, 16000]),
    )
    calls = []
    original = ReferenceCondition.decoder_kwargs

    def record_kwargs(condition, batch_size, **kwargs):
        calls.append(batch_size)
        return original(condition, batch_size, **kwargs)

    monkeypatch.setattr(ReferenceCondition, "decoder_kwargs", record_kwargs)
    features = target.conditioning.reference.features.clone()
    for _ in range(2):
        assert len(system.convert_batch(batch, target=target)) == 2
        torch.testing.assert_close(target.conditioning.reference.features, features)
    assert calls == [2, 2]


def test_source_voice_conversion_uses_source_reconstruction():
    from quick_convert.systems.conversion import SourceVoice

    system = SSLReconstructionConversionSystem(
        decoder=FakeDecoder(), online_encoders={"content": FakeContentEncoder(), "speaker": FakeSpeakerEncoder()}
    ).eval()
    batch = make_audio_batch()
    target = system.prepare_target(SourceVoice())
    generated = system.convert_batch(batch, target=target)
    expected = system(batch).generation.audio
    torch.testing.assert_close(generated.waveforms, expected.waveforms)
    torch.testing.assert_close(generated.lengths, expected.lengths)
    with pytest.raises(ValueError, match="inference_cfg_rate"):
        system.convert_batch(batch, target=target, inference_cfg_rate=1)

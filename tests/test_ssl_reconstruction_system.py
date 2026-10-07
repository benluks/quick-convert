from pathlib import Path

import pytest
import torch
from torch import nn

from quick_convert.components.decoders import CosyVoiceGenerationOutput
from quick_convert.components.layers.rvq import RVQLosses, RVQOutput
from quick_convert.components.mixins.resource import ResolvedResource
from quick_convert.components.ssl.base import ContentFeatures
from quick_convert.data import AudioBatch, GeneratedAudio
from quick_convert.systems.reconstruction import SSLReconstructionResult, SSLReconstructionSystem


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
    return SSLReconstructionSystem(
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
    system = SSLReconstructionSystem(
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
    system = SSLReconstructionSystem(
        decoder=FakeDecoder(),
        online_encoders={"content": encoder},
    )

    resource = system.get_resource(make_audio_batch(), "content")

    assert not encoder.scale.requires_grad
    assert not resource.values.requires_grad


@pytest.mark.parametrize("wrapped", [False, True])
def test_trainable_online_encoder_preserves_gradients_and_updates(wrapped):
    encoder = TrainableContentEncoder(wrapped=wrapped)
    system = SSLReconstructionSystem(
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

    optimizer.step()

    assert not torch.equal(before, encoder.scale.detach())


def test_precomputed_resources_remain_detached_with_trainable_online_encoder():
    encoder = TrainableContentEncoder()
    system = SSLReconstructionSystem(
        decoder=FakeDecoder(),
        online_encoders={"content": encoder},
        trainable_online_encoders=("content",),
    )
    batch = make_audio_batch()
    source = torch.ones(1, 3, 4, requires_grad=True)
    batch.resources["content"] = ResolvedResource(source, torch.tensor([3]))

    resource = system.get_resource(batch, "content")

    assert not resource.values.requires_grad

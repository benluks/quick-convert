from pathlib import Path

import torch
from torch import nn

from quick_convert.components.decoders import CosyVoiceGenerationOutput
from quick_convert.components.layers.rvq import RVQLosses, RVQOutput
from quick_convert.components.mixins.resource import ResolvedResource
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

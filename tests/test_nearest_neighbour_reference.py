from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from quick_convert.components.conditioning.nearest_neighbour import CosineTopK, NearestNeighbourMatching
from quick_convert.components.conditioning.nearest_neighbour_reference import NearestNeighbourReferenceConditioning
from quick_convert.components.resources.resolution import ResolvedResource
from quick_convert.data import AudioBatch, GeneratedAudio
from quick_convert.systems.conversion.ssl_reconstruction import SSLReconstructionConversionSystem
from quick_convert.systems.conversion.targets import ReferenceSpeech


class Decoder(nn.Module):
    requires_speaker_embedding = False

    def generate(self, features, lengths, speaker_embedding=None, **kwargs):
        return SimpleNamespace(audio=GeneratedAudio(features[..., 0], lengths, 16000))


def test_reference_pool_reuse_and_padding(monkeypatch):
    encoder = nn.Identity()
    encoder.sample_rate = 16000
    system = SSLReconstructionConversionSystem(
        decoder=Decoder(),
        online_encoders={"content": encoder},
        reference_conditioning=NearestNeighbourReferenceConditioning(NearestNeighbourMatching(CosineTopK(k=1))),
    ).eval()
    calls = []

    def load(path, **kwargs):
        calls.append(path)
        return torch.tensor([1.0 if path.name == "a.wav" else 2.0])

    def resource(batch, name):
        if name in batch.resources:
            return batch.resources[name]
        value = batch.waveforms[0, 0]
        frames = torch.tensor([[[value, 0.0], [0.0, value], [999.0, 999.0]]])
        return ResolvedResource(values=frames, lengths=torch.tensor([2]))

    monkeypatch.setattr("quick_convert.components.conditioning.nearest_neighbour_reference.load_audio_input", load)
    monkeypatch.setattr(system, "get_resource", resource)
    target = system.prepare_target(ReferenceSpeech(["a.wav", "b.wav"]))
    assert target.conditioning.shape == (4, 2)
    batch = AudioBatch(
        utt_ids=["x", "y"],
        paths=[Path("x.wav"), Path("y.wav")],
        splits=[None, None],
        resources={
            "content": ResolvedResource(
                values=torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [999.0, 999.0]]]),
                lengths=torch.tensor([2, 1]),
            )
        },
        waveforms=torch.ones(2, 2),
        lengths=torch.tensor([2, 1]),
        sample_rates=torch.tensor([16000, 16000]),
    )
    for _ in range(2):
        audio = system.convert_batch(batch, target=target)
        assert audio.lengths.tolist() == [2, 1]
        assert audio.waveforms[0, 0] in (1.0, 2.0)
        assert audio.waveforms[:, 1].eq(0).all()
    assert len(calls) == 2
    other = SSLReconstructionConversionSystem(
        decoder=Decoder(), reference_conditioning=NearestNeighbourReferenceConditioning()
    ).eval()
    with pytest.raises(ValueError, match="incompatible"):
        other.convert_batch(batch, target=target)

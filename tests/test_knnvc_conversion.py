from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_grouped_conversion import sample

from quick_convert.data import AudioBatch
from quick_convert.systems.conversion.knnvc import KNNVCConversionSystem
from quick_convert.systems.conversion.targets import ReferenceSpeech


def test_knn_conversion_preserves_prepared_targets(monkeypatch):
    def match(query, matching_set, **kwargs):
        matching_set.zero_()
        return torch.ones(5)

    model = SimpleNamespace(
        get_matching_set=lambda paths: torch.ones(3, 4),
        get_features=lambda waveform: torch.ones(3, 4),
        match=match,
    )
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: model)
    system = KNNVCConversionSystem()
    system.device = torch.device("cpu")
    target = system.prepare_target(ReferenceSpeech(["reference.wav"]))
    result = system.convert_batch(AudioBatch.from_samples([sample("source", "a")]), target=target)
    assert result.lengths.tolist() == [5]
    assert torch.all(target.conditioning == 1)
    assert not hasattr(system, "matching_set")
    with pytest.raises(ValueError, match="incompatible"):
        system.convert_batch(AudioBatch.from_samples([sample("source", "a")]), target=replace(target, space="other"))
    with pytest.raises(ValueError, match="does not support"):
        system.prepare_target(ReferenceSpeech(["reference.wav"], prompt_seconds=1))


def test_knn_targets_interleave_and_source_vocoder_has_batch_axis(monkeypatch):
    from quick_convert.systems.conversion import SourceVoice

    seen = []

    def match(query, matching_set, **kwargs):
        seen.append(float(matching_set[0, 0]))
        matching_set.zero_()
        return torch.ones(5)

    def vocode(features):
        assert features.shape == (1, 3, 4)
        return torch.ones(1, 1, 7)

    values = iter([1.0, 2.0])
    model = SimpleNamespace(
        get_matching_set=lambda paths: torch.full((3, 4), next(values)),
        get_features=lambda waveform: torch.ones(3, 4),
        match=match,
        vocode=vocode,
    )
    monkeypatch.setattr(torch.hub, "load", lambda *a, **kw: model)
    system = KNNVCConversionSystem(device="cpu")
    first = system.prepare_target(ReferenceSpeech(["a.wav"]))
    second = system.prepare_target(ReferenceSpeech(["b.wav"]))
    batch = AudioBatch.from_samples([sample("source", "a")])
    for target in (first, second, first):
        system.convert_batch(batch, target=target)
    assert seen == [1, 2, 1]
    assert system.convert_batch(batch, target=system.prepare_target(SourceVoice())).lengths.tolist() == [7]

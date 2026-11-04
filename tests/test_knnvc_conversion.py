from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_grouped_conversion import sample

from quick_convert.data import AudioBatch
from quick_convert.systems.anonymization.knnvc import KNNVCAnonymizer
from quick_convert.systems.anonymization.targets import ReferenceSpeech


def test_knn_conversion_preserves_prepared_and_legacy_targets(monkeypatch):
    def match(query, matching_set, **kwargs):
        matching_set.zero_()
        return torch.ones(5)

    model = SimpleNamespace(
        get_matching_set=lambda paths: torch.ones(3, 4),
        get_features=lambda waveform: torch.ones(3, 4),
        match=match,
    )
    monkeypatch.setattr(torch.hub, "load", lambda *args, **kwargs: model)
    system = KNNVCAnonymizer()
    system.device = torch.device("cpu")
    system.matching_set = torch.full((3, 4), 7.0)
    target = system.prepare_target(ReferenceSpeech(["reference.wav"]))
    result = system.convert_batch(AudioBatch.from_samples([sample("source", "a")]), target=target)
    assert result.lengths.tolist() == [5]
    assert torch.all(target.conditioning == 1)
    assert torch.all(system.matching_set == 7)
    with pytest.raises(ValueError, match="incompatible"):
        system.convert_batch(AudioBatch.from_samples([sample("source", "a")]), target=replace(target, space="other"))
    with pytest.raises(ValueError, match="does not support"):
        system.prepare_target(ReferenceSpeech(["reference.wav"], prompt_seconds=1))

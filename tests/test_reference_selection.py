import random
from pathlib import Path

import pytest
import soundfile as sf
import torch

from quick_convert.pipelines.targets import select_reference_paths


def test_duration_budget_keeps_whole_utterances_and_combines_with_count_limit(tmp_path):
    paths = []
    for index in range(3):
        path = tmp_path / f"{index}.wav"
        sf.write(path, torch.zeros(4000).numpy(), 1000)
        paths.append(path)
    assert select_reference_paths(paths, max_seconds=5) == paths[:2]
    assert select_reference_paths(paths, max_seconds=5, max_utterances=1) == paths[:1]
    assert select_reference_paths(paths, max_seconds=100) == paths


def test_count_selection_is_reproducible_and_does_not_read_audio(monkeypatch):
    paths = [Path(f"{i:02}.wav") for i in range(20)]

    def unexpected(*args, **kwargs):
        raise AssertionError("Count-only selection must not read audio headers")

    monkeypatch.setattr(sf, "info", unexpected)
    state = random.getstate()
    a = select_reference_paths(paths, max_utterances=4, shuffle=True, seed=52)
    assert a == select_reference_paths(list(reversed(paths)), max_utterances=4, shuffle=True, seed=52)
    assert a != select_reference_paths(paths, max_utterances=4, shuffle=True, seed=53)
    assert random.getstate() == state
    assert select_reference_paths(paths, max_utterances=2) == paths[:2]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_utterances": 0},
        {"max_utterances": True},
        {"max_seconds": 0},
        {"max_seconds": float("nan")},
        {"max_seconds": float("inf")},
        {"seed": "0"},
    ],
)
def test_invalid_selection_policy_fails_even_for_empty_catalogue(kwargs):
    with pytest.raises(ValueError):
        select_reference_paths([], **kwargs)

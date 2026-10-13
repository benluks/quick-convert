import pytest
import torch

from quick_convert.components.prosody import ProsodyTransform


def records():
    for index in range(12):
        pitch = torch.linspace(80 + index, 200 + index, 50, dtype=torch.float64)
        voiced = torch.arange(50) % 3 != 0
        values = torch.stack(
            (torch.arange(50) * 0.01, pitch.masked_fill(~voiced, 0), voiced, torch.linspace(0.01, 0.3, 50), voiced),
            dim=-1,
        )
        yield values, str(index % 2)


@pytest.mark.parametrize("strategy", ["quantile", "equal_width"])
def test_streamed_statistics_and_small_corpus_edges_match_exact_reference(strategy):
    data = list(records())
    transform = ProsodyTransform(bins=strategy, quantile_sample_size=1000).fit(data)
    pitch = torch.cat([values[values[:, 2].bool(), 1].log() for values, _ in data])
    energy = torch.cat([values[:, 3].log() for values, _ in data])
    for name, values in [("pitch", pitch), ("energy", energy)]:
        torch.testing.assert_close(transform.state[name], torch.stack((values.mean(), values.std(correction=0))))
        results = [transform.transform(record) for record in data]
        normalized = torch.cat([r[name][r["voiced"]] if name == "pitch" else r[name] for r in results])
        if strategy == "quantile":
            expected = torch.quantile(normalized, torch.arange(1, 32, dtype=torch.float64) / 32)
        else:
            expected = torch.linspace(normalized.min(), normalized.max(), 33, dtype=torch.float64)[1:-1]
        torch.testing.assert_close(transform.state[name + "_edges"], torch.unique_consecutive(expected))


def test_bounded_sampling_reproducible_and_one_shot_matches_replay():
    data = list(records())
    first = ProsodyTransform(quantile_sample_size=32, seed=17).fit(data)
    replay = ProsodyTransform(quantile_sample_size=32, seed=17).fit(iter(data))
    assert first.state["fitting"]["sampled_frames"] == {"pitch": 32, "energy": 32}
    assert first.state["fitting"]["frames"] == {"pitch": 396, "energy": 600}
    for name in ("pitch", "energy", "pitch_edges", "energy_edges"):
        torch.testing.assert_close(first.state[name], replay.state[name])
    different = ProsodyTransform(quantile_sample_size=32, seed=18).fit(data)
    assert not torch.equal(first.state["pitch_edges"], different.state["pitch_edges"])


def test_previous_statistics_load_without_sampling_config(tmp_path):
    transform = ProsodyTransform().fit(list(records()))
    state = transform.state_dict()
    state = {
        **state,
        "config": {key: value for key, value in state["config"].items() if key not in {"seed", "quantile_sample_size"}},
    }
    path = tmp_path / "old.pt"
    torch.save(state, path)
    restored = ProsodyTransform.load(path)
    values, speaker = next(records())
    torch.testing.assert_close(
        restored.transform(values, speaker)["pitch"], transform.transform(values, speaker)["pitch"]
    )


def test_invalid_sample_capacity():
    with pytest.raises(ValueError, match="positive integer"):
        ProsodyTransform(quantile_sample_size=0)

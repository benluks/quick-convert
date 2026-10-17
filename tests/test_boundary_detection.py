import pytest
import torch

from quick_convert.components.boundaries import change_signal, detect_boundaries, segment_measurements
from quick_convert.utils.streaming_statistics import RunningMoments


def test_change_metrics():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    torch.testing.assert_close(change_signal(x, "l2"), torch.tensor([2**0.5, 1.0]))
    torch.testing.assert_close(change_signal(x, "cosine"), torch.tensor([1.0, 0.0]))
    with pytest.raises(ValueError):
        change_signal(x, "invalid")


def test_peak_distance_and_frame_positions():
    signal = torch.tensor([0.0, 4.0, 0.0, 3.0, 0.0, 7.0, 0.0])
    assert detect_boundaries(signal, threshold=2).tolist() == [2, 4, 6]
    assert detect_boundaries(signal, threshold=2, min_distance_frames=3).tolist() == [2, 6]
    assert detect_boundaries(signal, threshold=8).numel() == 0


def test_segment_durations_and_edges():
    segments = segment_measurements(torch.tensor([2, 5]), num_frames=8, frame_hop_seconds=0.02)
    assert [round(x["duration_ms"]) for x in segments] == [40, 60, 60]
    assert [x["is_edge"] for x in segments] == [True, False, True]
    assert (
        len(segment_measurements(torch.tensor([2, 5]), num_frames=8, frame_hop_seconds=0.02, include_edges=False)) == 1
    )


def test_moments_batch_and_merge_equivalence():
    values = torch.arange(101, dtype=torch.float64) / 7
    streamed, merged = RunningMoments(), RunningMoments()
    for chunk in values.split(13):
        streamed.update(chunk)
        partial = RunningMoments()
        partial.update(chunk)
        merged.merge(partial)
    for result in (streamed, merged):
        assert result.count == len(values)
        assert result.mean == pytest.approx(values.mean().item())
        assert result.std == pytest.approx(values.std(correction=0).item())


def test_invalid_boundaries():
    with pytest.raises(ValueError):
        segment_measurements(torch.tensor([4, 2]), num_frames=8, frame_hop_seconds=0.02)

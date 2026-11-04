import pytest
import torch

from quick_convert.components.layers.heads import HeadTarget
from quick_convert.components.metrics import FramePredictionMetrics


def test_bins_ignore_unobserved_and_padding_and_merge_counts():
    metric = FramePredictionMetrics("bins", 3, "cpu")
    target = HeadTarget(torch.tensor([[0, 2, 999, 999]]), torch.tensor([3]), torch.tensor([[True, True, False, True]]))
    logits = torch.tensor([[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 3.0]]])
    metric.update(logits, target, torch.ones(1, 4, dtype=torch.bool))
    other = FramePredictionMetrics("bins", 3, "cpu")
    other.update(torch.tensor([[[0.0, 0.0, 3.0]]]), HeadTarget(torch.tensor([[2]])), torch.ones(1, 1, dtype=torch.bool))
    metric.merge(other)
    result = metric.compute()
    assert result["valid_frames"] == 3
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["mae_bins"] == pytest.approx(1 / 3)
    assert result["majority_baseline_accuracy"] == pytest.approx(2 / 3)
    assert result["recall/bin_2"] == 0.5
    assert "recall/bin_1" not in result


def test_voicing_f1_uses_global_counts_and_zero_logit_is_voiced():
    metric = FramePredictionMetrics("binary", 2, "cpu")
    metric.update(
        torch.tensor([[[0.0], [-1.0], [1.0]]]),
        HeadTarget(torch.tensor([[[1.0], [1.0], [0.0]]])),
        torch.ones(1, 3, dtype=torch.bool),
    )
    result = metric.compute()
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5
    assert result["predicted_voiced_fraction"] == pytest.approx(2 / 3)


def test_regression_empty_and_nan_exclusions():
    metric = FramePredictionMetrics("regression", 0, "cpu")
    assert set(metric.compute()) == {"valid_frames"}
    metric.update(
        torch.tensor([[[3.0], [float("nan")]]]),
        HeadTarget(torch.tensor([[[1.0], [float("nan")]]]), observation_mask=torch.tensor([[True, False]])),
        torch.ones(1, 2, dtype=torch.bool),
    )
    assert metric.compute()["mae"] == 2
    assert metric.compute()["rmse"] == 2


def test_compute_reduces_a_snapshot_and_reset_clears_counts(monkeypatch):
    metric = FramePredictionMetrics("binary", 2, "cpu")
    metric.update(torch.ones(1, 1, 1), HeadTarget(torch.ones(1, 1, 1)), torch.ones(1, 1, dtype=torch.bool))
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda state: state.mul_(2))
    assert metric.compute(sync_dist=True)["valid_frames"] == 2
    assert metric.compute(sync_dist=True)["valid_frames"] == 2
    assert metric.compute()["valid_frames"] == 1
    metric.reset()
    assert metric.compute()["valid_frames"] == 0


def test_merge_rejects_incompatible_contracts():
    with pytest.raises(ValueError, match="different prediction contracts"):
        FramePredictionMetrics("bins", 3, "cpu").merge(FramePredictionMetrics("binary", 2, "cpu"))


@pytest.mark.parametrize("values", [torch.tensor([[2]]), torch.tensor([[1.5]]), torch.tensor([[True]])])
def test_invalid_bin_targets_do_not_change_state(values):
    metric = FramePredictionMetrics("bins", 2, "cpu")
    with pytest.raises(ValueError, match="integer class indices"):
        metric.update(torch.ones(1, 1, 2), HeadTarget(values), torch.ones(1, 1, dtype=torch.bool))
    assert metric.compute()["valid_frames"] == 0


@pytest.mark.parametrize("kind", ["binary", "regression"])
def test_single_frame_without_trailing_dimension(kind):
    metric = FramePredictionMetrics(kind, 2 if kind == "binary" else 0, "cpu")
    metric.update(torch.ones(1, 1), HeadTarget(torch.ones(1, 1)), torch.ones(1, 1, dtype=torch.bool))
    assert metric.compute()["valid_frames"] == 1
    assert metric.compute()["mae"] == 0


def test_observed_nan_and_nonbinary_targets_are_rejected():
    metric = FramePredictionMetrics("binary", 2, "cpu")
    mask = torch.ones(1, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="zero or one"):
        metric.update(torch.ones(1, 1), HeadTarget(torch.full((1, 1), 0.5)), mask)
    with pytest.raises(ValueError, match="finite"):
        metric.update(torch.full((1, 1), float("nan")), HeadTarget(torch.ones(1, 1)), mask)


def test_target_validity_preserves_masks_and_rejects_bad_lengths():
    validity = torch.ones(1, 3, 1, dtype=torch.bool)
    target = HeadTarget(torch.ones(1, 3), torch.tensor([2]), torch.tensor([[True, False, True]]))
    assert target.valid_mask(validity).tolist() == [[True, False, False]]
    assert validity.all()
    target.lengths = torch.tensor([True])
    with pytest.raises(ValueError, match="integer"):
        target.valid_mask(validity)

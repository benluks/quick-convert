import json
from types import SimpleNamespace

import pytest
import torch

from quick_convert.pipelines.evaluation.metrics.eer import compute_eer
from quick_convert.pipelines.evaluation.protocols.asv import ASVEvaluator
from quick_convert.pipelines.evaluation.protocols.asv.trials import ASVTrial


class Dataset:
    def __init__(self, rows):
        self.rows = [
            SimpleNamespace(utt_id=uid, resources={"spkid": SimpleNamespace(value=spk)}, vector=torch.tensor(vector))
            for uid, spk, vector in rows
        ]

    def make_dataloader(self, **kwargs):
        return [self.rows] if self.rows else []


class Encoder:
    def __init__(self):
        self.calls = 0

    def eval(self):
        return self

    def encode_batch(self, batch):
        self.calls += 1
        return SimpleNamespace(values=torch.stack([row.vector for row in batch]).float())


def make_evaluator(tmp_path, **kwargs):
    enroll = Dataset([("a1", "a", [1.0, 0.0]), ("a2", "a", [1.0, 0.0]), ("b1", "b", [0.0, 1.0])])
    test = Dataset([("a3", "a", [1.0, 0.0]), ("b2", "b", [0.0, 1.0])])
    return ASVEvaluator(enroll, test, Encoder(), tmp_path, **kwargs)


def test_asv_all_pairs_and_saved_results(tmp_path):
    evaluator = make_evaluator(tmp_path)
    result = evaluator.run()
    assert result == {"eer": 0.0, "eer_threshold": 1.0, "num_target_trials": 2, "num_non_target_trials": 2}
    assert json.loads((tmp_path / "results.json").read_text()) == result


def test_explicit_trials(tmp_path):
    evaluator = make_evaluator(tmp_path)
    result = evaluator.evaluate(
        evaluator.enroll_dataset, evaluator.test_dataset, [ASVTrial("a", "a3", True), ASVTrial("b", "a3", False)]
    )
    assert result["eer"] == 0
    with pytest.raises(ValueError, match="Unknown"):
        evaluator.evaluate(evaluator.enroll_dataset, evaluator.test_dataset, [ASVTrial("missing", "a3", True)])


def test_cache_requires_identity_and_rejects_changed_key(tmp_path):
    with pytest.raises(ValueError, match="cache_key"):
        make_evaluator(tmp_path, cache_embeddings=True)
    first = make_evaluator(tmp_path, cache_embeddings=True, cache_key="model-data-v1")
    expected = first.run()
    second = make_evaluator(tmp_path, cache_embeddings=True, cache_key="model-data-v1")
    assert second.run() == expected
    assert second.speaker_encoder.calls == 0
    third = make_evaluator(tmp_path, cache_embeddings=True, cache_key="model-data-v2")
    with pytest.raises(ValueError, match="cache_key changed"):
        third.run()


def test_empty_duplicates_and_one_trial_class_are_rejected(tmp_path):
    evaluator = make_evaluator(tmp_path)
    with pytest.raises(ValueError, match="nonempty"):
        evaluator.evaluate(Dataset([]), evaluator.test_dataset)
    with pytest.raises(ValueError, match="both target and non-target"):
        evaluator.evaluate(evaluator.enroll_dataset, evaluator.test_dataset, [ASVTrial("a", "a3", True)])
    duplicate = Dataset([("same", "a", [1.0, 0.0]), ("same", "a", [1.0, 0.0])])
    with pytest.raises(ValueError, match="Duplicate"):
        evaluator.extract_embeddings(duplicate)


@pytest.mark.parametrize("bad", [[0.0, 0.0], [float("nan"), 1.0]])
def test_invalid_embeddings_are_rejected(tmp_path, bad):
    with pytest.raises(ValueError, match="finite, nonzero"):
        make_evaluator(tmp_path).extract_embeddings(Dataset([("bad", "a", bad)]))


def test_eer_ties_and_invalid_inputs():
    assert compute_eer(torch.ones(3), torch.ones(2)) == (0.5, 1.0)
    with pytest.raises(ValueError, match="empty"):
        compute_eer(torch.tensor([]), torch.ones(2))
    with pytest.raises(ValueError, match="finite"):
        compute_eer(torch.tensor([float("inf")]), torch.ones(2))

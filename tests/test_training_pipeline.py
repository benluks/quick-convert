from pathlib import Path
from types import SimpleNamespace

import pytest

from quick_convert.pipelines.training.pipeline import TrainingPipeline
from quick_convert.training.tokenizer import TokenizerTrainer


class RecordingTrainer:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.events = []

    def prepare(self, train_dataset, out_dir=None):
        self.events.append(("prepare", train_dataset, out_dir))
        return self.output_dir

    def train(self, train_dataset, val_dataset=None):
        self.events.append(("train", train_dataset, val_dataset))
        return "trained"


def make_pipeline(tmp_path: Path):
    trainer = RecordingTrainer(tmp_path / "run")
    pipeline = TrainingPipeline(
        trainer=trainer,
        train_dataset="train",
        val_dataset="validation",
        out_dir=tmp_path / "configured",
    )
    return pipeline, trainer


def test_construction_does_not_prepare_the_backend(tmp_path):
    _, trainer = make_pipeline(tmp_path)

    assert trainer.events == []


def test_prepare_is_explicit_and_idempotent(tmp_path):
    pipeline, trainer = make_pipeline(tmp_path)

    assert pipeline.prepare() == tmp_path / "run"
    assert pipeline.prepare() == tmp_path / "run"

    assert trainer.events == [("prepare", "train", tmp_path / "configured")]


def test_run_prepares_direct_library_usage(tmp_path):
    pipeline, trainer = make_pipeline(tmp_path)

    assert pipeline.run() == "trained"
    assert trainer.events == [
        ("prepare", "train", tmp_path / "configured"),
        ("train", "train", "validation"),
    ]


def test_config_is_written_to_the_prepared_run_directory(tmp_path):
    pipeline, _ = make_pipeline(tmp_path)

    with pytest.raises(RuntimeError, match="Prepare"):
        pipeline.write_config("before: prepare")

    pipeline.prepare()
    pipeline.write_config("ready: true\n")

    assert (tmp_path / "run" / "config.yaml").read_text() == "ready: true\n"


def test_tokenizer_backend_uses_the_shared_prepare_contract(tmp_path):
    class TokenizerModule:
        def train_from_iterator(self, sentences, output_dir, model_prefix):
            assert list(sentences) == ["hello", "world"]
            assert output_dir == tmp_path / "tokenizer"
            assert model_prefix == "tokens"
            return output_dir / f"{model_prefix}.model"

    dataset = [
        SimpleNamespace(resources={"transcript": SimpleNamespace(value="hello")}),
        SimpleNamespace(resources={"transcript": SimpleNamespace(value="world")}),
    ]
    pipeline = TrainingPipeline(
        trainer=TokenizerTrainer(module=TokenizerModule(), model_prefix="tokens"),
        train_dataset=dataset,
        out_dir=tmp_path / "tokenizer",
    )

    assert pipeline.run() == tmp_path / "tokenizer" / "tokens.model"

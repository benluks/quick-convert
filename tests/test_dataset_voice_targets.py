from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from quick_convert.data import ManifestDataset
from quick_convert.pipelines.conversion import reference_speech_from_dataset
from quick_convert.utils.resolvers import register_config_resolvers


def test_librispeech_target_uses_resources_across_splits_without_loading_audio(tmp_path):
    for split, speaker, utterance in [
        ("test-clean", "42", "42-1-1"),
        ("train-clean-100", "42", "42-2-1"),
        ("test-clean", "7", "7-1-1"),
    ]:
        path = tmp_path / split / speaker / "1" / f"{utterance}.flac"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()  # Not decodable: selection must use metadata only.
    register_config_resolvers()
    config_dir = Path(__file__).parents[1] / "quick_convert/configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="run/convert_reference_librispeech",
            overrides=[
                "source=/unused/run",
                "target_speaker=42",
                f"dataset.root={tmp_path}",
                f"target_dataset.root={tmp_path}",
                "target_dataset.splits=[test-clean,train-clean-100]",
            ],
        )
    target = instantiate(cfg.pipeline.target)
    assert {p.stem for p in target.paths} == {"42-1-1", "42-2-1"}
    assert target.aggregation is None
    source_dataset = instantiate(cfg.dataset)
    assert {row.utt_id for row in source_dataset.rows} == {"42-1-1", "7-1-1"}


def test_manifest_target_selection_and_missing_speaker(tmp_path):
    manifest = tmp_path / "references.csv"
    manifest.write_text("utt_id,path,identity\na,/data/a.wav,42\nb,/data/b.wav,7\n")
    dataset = ManifestDataset(manifest, resources={"speaker": {"column": "identity", "kind": "text"}})
    target = reference_speech_from_dataset(dataset, 42, speaker_resource="speaker")
    assert target.paths == (Path("/data/a.wav"),)
    with pytest.raises(ValueError, match="No recordings"):
        reference_speech_from_dataset(dataset, "99", speaker_resource="speaker")
    with pytest.raises(ValueError, match="lacks speaker resource"):
        reference_speech_from_dataset(dataset, 42)


def test_random_librispeech_config_uses_source_and_target_speaker_resources(tmp_path):
    from quick_convert.pipelines.targets import DatasetReferenceTargets

    for speaker in ("42", "7"):
        path = tmp_path / "test-clean" / speaker / "1" / f"{speaker}-1-1.flac"
        path.parent.mkdir(parents=True)
        path.touch()
    register_config_resolvers()
    config_dir = Path(__file__).parents[1] / "quick_convert/configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="run/convert_reference_librispeech_random",
            overrides=[
                "source=/unused/run",
                f"dataset.root={tmp_path}",
                f"target_dataset.root={tmp_path}",
                "target_dataset.splits=[test-clean]",
                "+pipeline.target.seed=42",
            ],
        )
    target = instantiate(cfg.pipeline.target)
    assert isinstance(target, DatasetReferenceTargets)
    assert target.seed == 42 and target.exclude_source
    assert target.groups["42"].aggregation is None
    assert target.groups["42"].prompt_seconds == 3.0
    cfg.dataset.load = ["spkid"]  # Selection must not decode the dummy audio.
    dataset = instantiate(cfg.dataset)
    for source in dataset:
        assert target.select(source) != source.resources.spkid.value

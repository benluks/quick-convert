from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


@pytest.mark.parametrize("speaker,expected", [(None, {"1089-1-1", "121-1-1"}), ("1089", {"1089-1-1"})])
def test_knnvc_librispeech_selects_sources_and_prepares_reference_paths(tmp_path, speaker, expected):
    for split, identity in [("test-clean", "1089"), ("test-clean", "121"), ("train-clean-100", "1069")]:
        path = tmp_path / split / identity / "1" / f"{identity}-1-1.flac"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    overrides = [f"root={tmp_path}", "target_speaker=1069", "target_max_seconds=null"]
    if speaker is not None:
        overrides.append(f"source_speaker={speaker}")
    with initialize_config_dir(config_dir=str(Path("quick_convert/configs").resolve()), version_base=None):
        cfg = compose(config_name="run/convert_knnvc_librispeech", overrides=overrides)
    dataset = instantiate(cfg.dataset)
    assert {row.utt_id for row in dataset.rows} == expected
    target = instantiate(cfg.target)
    assert [p.stem for p in target.paths] == ["1069-1-1"]
    assert target.prompt_seconds is None and target.aggregation is None


def test_convert_verb_selects_knnvc_librispeech():
    from main import _resolve_config

    config, overrides = _resolve_config(
        "convert", ["knnvc_librispeech", "root=/data/LibriSpeech"], Path("quick_convert/configs/run")
    )
    assert config == "run/convert_knnvc_librispeech"
    assert overrides == ["root=/data/LibriSpeech"]


def test_knnvc_reference_selection_is_independent_of_corpus_layout_and_format(tmp_path):
    from omegaconf import OmegaConf

    for speaker in ["alice", "bob"]:
        path = tmp_path / speaker / "reference.wav"
        path.parent.mkdir()
        path.touch()  # Selection must not decode audio.
    with initialize_config_dir(config_dir=str(Path("quick_convert/configs").resolve()), version_base=None):
        cfg = compose(
            config_name="run/convert_knnvc_clac", overrides=["target_speaker=alice", "target_max_seconds=null"]
        )
    cfg.target_speaker_resource = "performer"
    cfg.target_dataset = OmegaConf.create(
        {
            "_target_": "quick_convert.data.BaseDataset",
            "root": str(tmp_path),
            "file_format": "wav",
            "utt_id_template": "{path.parent.name}/{path.stem}",
            "load": ["performer"],
            "resource_providers": [
                {
                    "_target_": "quick_convert.data.resources.TemplateResourceProvider",
                    "name": "performer",
                    "kind": "text",
                    "template": "{path.parent.name}",
                }
            ],
        }
    )
    target = instantiate(cfg.target)
    assert target.paths == (tmp_path / "alice" / "reference.wav",)


def test_knnvc_default_budget_is_applied_before_target_preparation(tmp_path):
    import soundfile as sf
    import torch

    for index in range(20):
        path = tmp_path / "train-clean-100" / "1069" / "1" / f"1069-1-{index:02}.flac"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, torch.zeros(8000).numpy(), 1000)
    with initialize_config_dir(config_dir=str(Path("quick_convert/configs").resolve()), version_base=None):
        cfg = compose(
            config_name="run/convert_knnvc_librispeech", overrides=[f"root={tmp_path}", "target_speaker=1069"]
        )
    target = instantiate(cfg.target)  # No model construction or feature extraction.
    assert len(target.paths) == 8
    assert sum(sf.info(p).duration for p in target.paths) == 64
    assert target.paths == instantiate(cfg.target).paths
    assert cfg.target.max_utterances == 10
    assert cfg.target.shuffle is True and cfg.target.seed == 0

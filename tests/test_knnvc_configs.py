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
    overrides = [f"root={tmp_path}", "target_speaker=1069"]
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

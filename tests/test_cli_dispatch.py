from pathlib import Path

import pytest

from main import _resolve_config


def make_run_configs(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for name in ["anonymize_asrbn_clac", "eval_asr_librispeech", "train_vq_asr_librispeech"]:
        (run_dir / f"{name}.yaml").touch()
    return run_dir


def test_universal_command_accepts_full_run_config(tmp_path: Path) -> None:
    run_dir = make_run_configs(tmp_path)

    config, overrides = _resolve_config(
        "quick-convert",
        ["train_vq_asr_librispeech", "pipeline.batch_size=4"],
        run_dir,
    )

    assert config == "run/train_vq_asr_librispeech"
    assert overrides == ["pipeline.batch_size=4"]


def test_command_can_map_to_a_different_config_prefix(tmp_path: Path) -> None:
    run_dir = make_run_configs(tmp_path)

    config, overrides = _resolve_config("evaluate", ["asr_librispeech"], run_dir)

    assert config == "run/eval_asr_librispeech"
    assert overrides == []


def test_legacy_command_remains_compatible(tmp_path: Path) -> None:
    run_dir = make_run_configs(tmp_path)

    config, _ = _resolve_config("eval_asr", ["librispeech"], run_dir)

    assert config == "run/eval_asr_librispeech"


def test_missing_alias_lists_valid_choices(tmp_path: Path) -> None:
    run_dir = make_run_configs(tmp_path)

    with pytest.raises(SystemExit, match="asrbn_clac"):
        _resolve_config("anonymize", ["missing"], run_dir)

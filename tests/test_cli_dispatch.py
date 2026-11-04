import sys
from pathlib import Path

import pytest

from main import _resolve_config, main


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
        ["train_vq_asr_librispeech", "trainer.train_dataloader_kwargs.batch_size=4"],
        run_dir,
    )

    assert config == "run/train_vq_asr_librispeech"
    assert overrides == ["trainer.train_dataloader_kwargs.batch_size=4"]


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


def test_universal_help_lists_export(tmp_path: Path, capsys) -> None:
    run_dir = make_run_configs(tmp_path)

    with pytest.raises(SystemExit) as caught:
        _resolve_config("quick-convert", ["--help"], run_dir)
    assert caught.value.code == 0
    assert "quick-convert export <run-dir>" in capsys.readouterr().out


def test_missing_arguments_show_usage_and_fail(tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit) as caught:
        _resolve_config("quick-convert", [], make_run_configs(tmp_path))
    assert caught.value.code == 1
    assert "Usage: quick-convert" in capsys.readouterr().out


def test_universal_command_dispatches_export(monkeypatch) -> None:
    called_with = None

    class ExportModule:
        @staticmethod
        def main(argv):
            nonlocal called_with
            called_with = argv

    monkeypatch.setattr("main.importlib.import_module", lambda name: ExportModule)
    monkeypatch.setattr(sys, "argv", ["quick-convert", "export", "run", "model"])

    main()

    assert called_with == ["run", "model"]


@pytest.mark.parametrize("action", ["requirements", "doctor"])
def test_universal_command_dispatches_dependency_actions(monkeypatch, action) -> None:
    called_with = None

    class DependencyModule:
        @staticmethod
        def main(argv, *, action):
            nonlocal called_with
            called_with = (argv, action)
            return 0

    monkeypatch.setattr("main.importlib.import_module", lambda name: DependencyModule)
    monkeypatch.setattr(sys, "argv", ["quick-convert", action, "train_vq_asr_librispeech"])

    with pytest.raises(SystemExit, match="0"):
        main()

    assert called_with == (["train_vq_asr_librispeech"], action)


def test_run_dispatch_preserves_executable_for_distributed_relaunch(monkeypatch, tmp_path):
    from types import SimpleNamespace

    executable = tmp_path / "bin" / "quick-convert"
    seen = []
    monkeypatch.setattr(sys, "argv", [str(executable), "train_vq_asr_librispeech", "trainer.ddp.enabled=true"])
    monkeypatch.setattr(
        "main.importlib.import_module", lambda name: SimpleNamespace(main=lambda: seen.append(sys.argv[:]))
    )
    main()
    assert seen[0][0] == str(executable)
    assert seen[0][1] == "--config-path"
    assert seen[0][-1] == "trainer.ddp.enabled=true"
    # The child re-enters the console entry point with the expanded arguments.
    main()
    assert seen[1] == seen[0]

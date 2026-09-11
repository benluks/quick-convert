from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


COMMAND_PREFIXES = {
    "evaluate": "eval",
}


def _available_aliases(
    config_prefix: str,
    run_dir: Path = Path("configs/run"),
) -> list[str]:
    stem_prefix = f"{config_prefix}_"
    return sorted(
        path.stem[len(stem_prefix) :]
        for path in run_dir.glob(f"{stem_prefix}*.yaml")
        if path.stem.startswith(stem_prefix)
    )


def _available_run_configs(run_dir: Path) -> list[str]:
    return sorted(path.stem for path in run_dir.glob("*.yaml"))


def _resolve_config(
    command: str,
    argv: list[str],
    run_dir: Path,
) -> tuple[str, list[str]]:
    universal = command in {"quick-convert", "quick_convert"}
    config_prefix = COMMAND_PREFIXES.get(command, command)

    if not argv or argv[0] in {"-h", "--help"}:
        if universal:
            choices = ", ".join(_available_run_configs(run_dir)) or "(none found)"
            usage = f"Usage: {command} <run-config> [hydra overrides...]"
        else:
            choices = ", ".join(_available_aliases(config_prefix, run_dir)) or "(none found)"
            usage = f"Usage: {command} <config-alias> [hydra overrides...]"
        raise SystemExit(f"{usage}\nAvailable configurations: {choices}")

    name, *overrides = argv
    name = Path(name).stem
    config_stem = name if universal else f"{config_prefix}_{name}"
    config_file = run_dir / f"{config_stem}.yaml"

    if not config_file.is_file():
        choices = (
            _available_run_configs(run_dir)
            if universal
            else _available_aliases(config_prefix, run_dir)
        )
        raise SystemExit(
            f"No run configuration found at {config_file}.\n"
            f"Available configurations: {', '.join(choices) or '(none found)'}"
        )

    return f"run/{config_stem}", overrides


def main() -> None:
    command = Path(sys.argv[0]).stem
    run_dir = Path(__file__).resolve().parent / "configs" / "run"
    config_name, overrides = _resolve_config(command, sys.argv[1:], run_dir)
    module = importlib.import_module("quick_convert.cli.run")

    os.environ["HYDRA_FULL_ERROR"] = "1"

    sys.argv = [
        command,
        "--config-path",
        str(run_dir.parent),
        "--config-name",
        config_name,
        *overrides,
    ]

    module.main()

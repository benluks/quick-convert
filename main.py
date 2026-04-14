from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path


def _available_aliases(config_prefix: str, run_dir: Path) -> list[str]:
    stem_prefix = f"{config_prefix}_"
    return sorted(
        path.stem[len(stem_prefix) :]
        for path in run_dir.glob(f"{stem_prefix}*.yaml")
        if path.stem.startswith(stem_prefix)
    )


def main() -> None:
    command = Path(sys.argv[0]).stem
    config_prefix = command
    module_name = f"quick_convert.cli.{config_prefix}"
    run_dir = Path(__file__).resolve().parent / "configs" / "run"

    if importlib.util.find_spec(module_name) is None:
        raise SystemExit(f"No CLI module found for command {command!r}: {module_name}")

    argv = sys.argv[1:]

    if not argv or argv[0] in {"-h", "--help"}:
        aliases = (
            ", ".join(_available_aliases(config_prefix, run_dir)) or "(none found)"
        )
        raise SystemExit(
            f"Usage: {command} <config-alias> [hydra overrides...]\n"
            f"Resolved module: {module_name}\n"
            f"Config prefix: {config_prefix}\n"
            f"Available config aliases: {aliases}"
        )

    config_alias, *overrides = argv
    config_name = f"run/{config_prefix}_{config_alias}"

    config_file = run_dir / f"{config_prefix}_{config_alias}.yaml"
    if not config_file.is_file():
        aliases = ", ".join(_available_aliases(config_prefix)) or "(none found)"
        raise SystemExit(
            f"No config found for alias {config_alias!r}.\n"
            f"Expected file: {config_file.name}\n"
            f"Available config aliases: {aliases}"
        )

    module = importlib.import_module(module_name)

    sys.argv = [
        command,
        "--config-name",
        config_name,
        *overrides,
    ]

    module.main()

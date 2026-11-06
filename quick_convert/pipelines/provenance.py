"""Run metadata used by conversion and evaluation workflows."""

import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml


def resolved_config(config: str | None) -> dict | None:
    """Read the resolved YAML supplied through the CLI's write_config hook."""
    return yaml.safe_load(config) if config is not None else None


def _git_output(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=Path(__file__).parents[2],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def execution_metadata() -> dict:
    """Describe the executing code and working directory when available."""
    revision = _git_output("rev-parse", "HEAD")
    status = _git_output("status", "--porcelain", "--untracked-files=no") if revision else None
    try:
        package_version = version("quick-convert")
    except PackageNotFoundError:
        package_version = None
    return {
        "git_commit": revision,
        "git_dirty": None if status is None else bool(status),
        "package_version": package_version,
        "working_directory": str(Path.cwd()),
    }

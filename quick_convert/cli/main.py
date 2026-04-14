# quick_convert/cli/main.py
from __future__ import annotations

import sys

from quick_convert.cli import anonymize as anonymize_cli


def anonymize_main() -> None:
    argv = sys.argv[1:]

    if not argv:
        raise SystemExit("Usage: anonymize <config-alias> [hydra overrides...]")

    config_alias, *overrides = argv
    config_name = f"run/anonymization_{config_alias}"

    sys.argv = [
        "anonymize",
        "--config-name",
        config_name,
        *overrides,
    ]

    anonymize_cli.main()

"""Inspect optional dependencies for a composed run."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from quick_convert.dependencies import DependencyPlan, compose_dependency_plan


def _parser(action: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"quick-convert {action}",
        description=f"{action.title()} optional dependencies for a composed run.",
    )
    parser.add_argument("run_config", help="Run config name, with or without the run/ prefix")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides applied before inspection")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Emit machine-readable JSON")
    return parser


def _payload(plan: DependencyPlan) -> dict:
    missing = plan.missing_extras()
    return {
        "extras": list(plan.extras),
        "missing": list(missing),
        "uv_command": plan.uv_command,
        "pip_specifier": plan.pip_specifier,
    }


def main(argv: Sequence[str] | None = None, *, action: str = "requirements") -> int:
    """Report or check dependencies for one composed run."""
    args = _parser(action).parse_args(argv)
    plan = compose_dependency_plan(args.run_config, args.overrides)
    payload = _payload(plan)

    if args.as_json:
        print(json.dumps(payload, indent=2))
    elif action == "requirements":
        print("Required extras: " + (", ".join(plan.extras) or "none"))
        print(f"Install from this checkout: {plan.uv_command}")
        print(f"Install from a package index: pip install '{plan.pip_specifier}'")
    else:
        missing = payload["missing"]
        if missing:
            print("Missing extras: " + ", ".join(missing))
            print(f"Install from this checkout: {plan.uv_command}")
        else:
            print("All optional dependencies for this run are available.")

    return 1 if action == "doctor" and payload["missing"] else 0

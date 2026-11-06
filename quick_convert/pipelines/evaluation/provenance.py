"""Metadata shared by the evaluation workflows."""

import json
from pathlib import Path


def describe_component(component) -> dict | None:
    if component is None:
        return None
    cls = type(component)
    description = {"type": f"{cls.__module__}.{cls.__qualname__}"}
    for key in ("name", "model_name", "root", "splits"):
        value = getattr(component, key, None)
        if value is not None:
            description[key] = value
    return description


def write_provenance(out_dir: Path, *, protocol: str, config: dict | None, **details) -> None:
    payload = {"schema_version": 1, "protocol": protocol, "resolved_config": config, **details}
    (out_dir / "provenance.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")

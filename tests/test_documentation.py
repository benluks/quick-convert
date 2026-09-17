from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).parents[1]
DOCS = ROOT / "docs"
RUN_CONFIGS = ROOT / "quick_convert" / "configs" / "run"


def _markdown_files():
    return [ROOT / "README.md", ROOT / "AGENTS.md", *DOCS.rglob("*.md")]


def test_relative_markdown_links_exist():
    missing = []
    for source in _markdown_files():
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", source.read_text()):
            target = target.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            if not (source.parent / target).resolve().exists():
                missing.append(f"{source.relative_to(ROOT)} -> {target}")

    assert not missing, "Missing documentation targets:\n" + "\n".join(missing)


def test_documented_workflow_runs_exist():
    guide = (DOCS / "guides" / "workflows.md").read_text()
    documented = set(re.findall(r"`(?:quick-convert )?([a-z]+_[a-z0-9_]+)(?:\s|`)", guide))
    available = {path.stem for path in RUN_CONFIGS.glob("*.yaml")}

    missing = sorted(documented - available)
    assert not missing, f"Documented run configs do not exist: {missing}"

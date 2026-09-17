from __future__ import annotations

import tomllib
from pathlib import Path

from omegaconf import OmegaConf

from quick_convert.dependencies import compose_dependency_plan


ROOT = Path(__file__).parents[1]


def test_vq_asr_requirements_follow_composed_targets():
    plan = compose_dependency_plan("train_vq_asr_librispeech")

    assert plan.extras == ("asr", "training", "transformers")
    assert plan.uv_command == "uv sync --extra asr --extra training --extra transformers"


def test_swapping_vq_asr_encoder_changes_requirements():
    plan = compose_dependency_plan(
        "train_vq_asr_librispeech",
        ["system.online_encoders.content._target_=quick_convert.components.ssl.DACContentEncoder"],
    )

    assert plan.extras == ("asr", "dac", "training")
    assert "transformers" not in plan.extras


def test_dependency_registry_only_uses_published_extras():
    registry = OmegaConf.load(ROOT / "quick_convert" / "configs" / "dependencies.yaml")
    with (ROOT / "pyproject.toml").open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)

    published = set(pyproject["project"]["optional-dependencies"])

    assert set(registry.extras) <= published
    assert all(set(extras) <= published for extras in registry.targets.values())

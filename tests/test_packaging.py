import tomllib
from pathlib import Path


PYPROJECT_PATH = Path(__file__).parents[1] / "pyproject.toml"


def test_uv_conflicts_only_reference_declared_extras():
    with PYPROJECT_PATH.open("rb") as file:
        pyproject = tomllib.load(file)

    extras = set(pyproject["project"]["optional-dependencies"])
    referenced_extras = {entry["extra"] for conflict in pyproject["tool"]["uv"]["conflicts"] for entry in conflict}

    assert referenced_extras <= extras


def test_asr_compatibility_extra_contains_atomic_asr_features():
    with PYPROJECT_PATH.open("rb") as file:
        pyproject = tomllib.load(file)

    extras = pyproject["project"]["optional-dependencies"]

    assert set(extras["asr"]) == set(extras["sentencepiece"]) | set(extras["wer"])


def test_lightning_compatibility_extra_matches_training_profile():
    with PYPROJECT_PATH.open("rb") as file:
        pyproject = tomllib.load(file)

    extras = pyproject["project"]["optional-dependencies"]

    assert set(extras["lightning"]) == set(extras["training"])

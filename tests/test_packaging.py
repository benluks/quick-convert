import tomllib
from pathlib import Path


PYPROJECT_PATH = Path(__file__).parents[1] / "pyproject.toml"


def test_uv_conflicts_only_reference_declared_extras():
    with PYPROJECT_PATH.open("rb") as file:
        pyproject = tomllib.load(file)

    extras = set(pyproject["project"]["optional-dependencies"])
    referenced_extras = {entry["extra"] for conflict in pyproject["tool"]["uv"]["conflicts"] for entry in conflict}

    assert referenced_extras <= extras

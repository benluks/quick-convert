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


def test_conditional_rvq_compatibility_extra_matches_emotion2vec():
    with PYPROJECT_PATH.open("rb") as file:
        pyproject = tomllib.load(file)

    extras = pyproject["project"]["optional-dependencies"]

    assert set(extras["conditional-rvq"]) == set(extras["emotion2vec"])


def test_logging_profiles_select_compatible_wandb_and_protobuf():
    with PYPROJECT_PATH.open("rb") as file:
        pyproject = tomllib.load(file)

    extras = pyproject["project"]["optional-dependencies"]
    assert "descript-audio-codec" in extras["dac"]
    for extra in ("dac", "training", "lightning"):
        assert "wandb==0.25.0" in extras[extra]
        assert "protobuf<7" in extras[extra]


def test_cosyvoice_extra_tracks_supported_runtime_surface():
    with PYPROJECT_PATH.open("rb") as file:
        pyproject = tomllib.load(file)

    requirements = pyproject["project"]["optional-dependencies"]["cosyvoice"]
    package_names = {requirement.split(">=")[0] for requirement in requirements}

    assert {"huggingface-hub", "scipy"} <= package_names
    assert {"gdown", "wget", "transformers", "lightning"}.isdisjoint(package_names)

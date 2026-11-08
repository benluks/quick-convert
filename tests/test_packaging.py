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


def test_dac_and_onnx_prefix_profiles_have_compatible_requirements():
    from packaging.requirements import Requirement

    with PYPROJECT_PATH.open("rb") as file:
        pyproject = tomllib.load(file)

    extras = pyproject["project"]["optional-dependencies"]
    requirements = {Requirement(value).name: Requirement(value) for value in extras["dac"]}
    assert set(requirements) == {"descript-audio-codec", "descript-audiotools", "wandb", "protobuf"}
    url = requirements["descript-audiotools"].url
    assert url.startswith("git+https://github.com/descriptinc/audiotools.git@")
    assert len(url.rsplit("@", 1)[1]) == 40
    assert "0.25.0" in requirements["wandb"].specifier
    assert "7.0.0" not in requirements["protobuf"].specifier
    onnx = next(Requirement(value) for value in extras["campplus"] if Requirement(value).name == "onnx")
    assert "1.18.0" in onnx.specifier and "1.19.0" not in onnx.specifier


def test_cosyvoice_extra_tracks_supported_runtime_surface():
    with PYPROJECT_PATH.open("rb") as file:
        pyproject = tomllib.load(file)

    requirements = pyproject["project"]["optional-dependencies"]["cosyvoice"]
    package_names = {requirement.split(">=")[0] for requirement in requirements}

    assert {"huggingface-hub", "scipy"} <= package_names
    assert {"gdown", "wget", "transformers", "lightning"}.isdisjoint(package_names)


def test_training_logging_profiles_select_compatible_wandb_and_protobuf():
    with PYPROJECT_PATH.open("rb") as file:
        extras = tomllib.load(file)["project"]["optional-dependencies"]
    for extra in ("dac", "training", "lightning"):
        assert "wandb==0.25.0" in extras[extra]
        assert "protobuf<7" in extras[extra]

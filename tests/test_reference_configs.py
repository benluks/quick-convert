from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


CONFIG_DIR = Path(__file__).parents[1] / "configs"


@pytest.fixture(autouse=True)
def register_resolvers():
    resolvers = {
        "add": lambda x, y: int(x) + int(y),
        "mul": lambda x, y: int(x) * int(y),
        "floor": lambda x, y: int(int(x) / int(y)),
        "bool": bool,
        "len": len,
    }
    for name, resolver in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)


@pytest.mark.parametrize(
    ("config_name", "pipeline_target"),
    [
        (
            "run/train_vq_asr_librispeech",
            "quick_convert.pipelines.training.pipeline.TrainingPipeline",
        ),
        (
            "run/train_sslr_w2vbert_cmdiff",
            "quick_convert.pipelines.training.pipeline.TrainingPipeline",
        ),
        (
            "run/build_manifest_libri",
            "quick_convert.pipelines.build_manifest.BuildManifestPipeline",
        ),
        (
            "run/precompute_content_w2vbert_librispeech",
            "quick_convert.pipelines.precompute_features.PrecomputeFeaturesPipeline",
        ),
        (
            "run/eval_asr_librispeech",
            "quick_convert.pipelines.evaluation.pipeline.EvalPipeline",
        ),
    ],
)
def test_reference_config_composes(config_name, pipeline_target):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        config = compose(config_name=config_name, return_hydra_config=True)

    assert config.pipeline._target_ == pipeline_target


def test_build_manifest_resource_providers_resolve_and_instantiate():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        config = compose(config_name="run/build_manifest_libri", return_hydra_config=True)

    providers = [instantiate(provider) for provider in config.dataset.resource_providers]

    assert [provider.name for provider in providers] == ["transcript", "spkid"]


def test_clac_root_comes_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("QUICK_CONVERT_CLAC_ROOT", str(tmp_path))

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        config = compose(config_name="dataset/clac")

    assert config.dataset.root == str(tmp_path)


def test_ssl_reconstruction_uses_ssl_features_directly():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        config = compose(
            config_name="run/train_sslr_w2vbert_cmdiff",
            return_hydra_config=True,
        )

    assert "encoder" not in config.architecture
    assert "encoder" not in config.trainer.module
    assert config.architecture.decoder.feature_dim == config.architecture.feature_dim

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
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
    ("config_name", "module_target"),
    [
        (
            "run/train_vq_asr_librispeech",
            "quick_convert.pipelines.training.modules.vq_asr.VQASRTrainingModule",
        ),
        (
            "run/train_sslr_w2vbert_cmdiff_rvq",
            "quick_convert.pipelines.training.modules.ssl_reconstruction.SSLReconstructionTrainingModule",
        ),
    ],
)
def test_reference_training_config_composes(config_name, module_target):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        config = compose(config_name=config_name, return_hydra_config=True)

    assert config.pipeline._target_ == "quick_convert.pipelines.training.pipeline.TrainingPipeline"
    assert config.trainer.module._target_ == module_target

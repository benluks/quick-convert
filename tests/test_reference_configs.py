import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from quick_convert.utils.resolvers import class_attribute


CONFIG_DIR = Path(__file__).parents[1] / "quick_convert" / "configs"
RUN_CONFIGS = sorted(path.stem for path in (CONFIG_DIR / "run").glob("*.yaml"))


@pytest.fixture(autouse=True)
def register_resolvers():
    resolvers = {
        "add": lambda x, y: int(x) + int(y),
        "mul": lambda x, y: int(x) * int(y),
        "floor": lambda x, y: int(int(x) / int(y)),
        "bool": bool,
        "len": len,
        "class_attr": class_attribute,
    }
    for name, resolver in resolvers.items():
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, resolver)


@pytest.mark.parametrize("run_config", RUN_CONFIGS)
def test_all_advertised_run_configs_compose(run_config):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        compose(config_name=f"run/{run_config}", return_hydra_config=True)


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
            "run/train_sslr_w2vbert_cmdiff_rvq",
            "quick_convert.pipelines.training.pipeline.TrainingPipeline",
        ),
        (
            "run/train_sslr_w2vbert_dit",
            "quick_convert.pipelines.training.pipeline.TrainingPipeline",
        ),
        (
            "run/train_bpe_tokenizer_librispeech",
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

    assert config.dataset._target_ == "quick_convert.data.BaseDataset"
    assert [provider.name for provider in providers] == ["transcript", "spkid"]


def test_librispeech_can_be_composed_into_a_named_dataset_slot():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        config = compose(
            config_name="run/train_vq_asr_librispeech",
            return_hydra_config=True,
        )

    assert config.source_dataset._target_ == "quick_convert.data.BaseDataset"


def test_tokenizer_training_has_an_explicit_output_directory():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        config = compose(
            config_name="run/train_bpe_tokenizer_librispeech",
            return_hydra_config=True,
        )

    assert config.pipeline.out_dir == "outputs/tokenizer/librispeech_1000_tokens"


def test_tokenizer_training_uses_the_transcript_provider():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        config = compose(
            config_name="run/train_bpe_tokenizer_librispeech",
            return_hydra_config=True,
        )

    providers = [instantiate(provider) for provider in config.train_dataset.resource_providers]

    assert [provider.name for provider in providers] == ["transcript"]


def test_flat_manifest_contains_vq_asr_resources():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        config = compose(
            config_name="run/build_flat_manifest_libri",
            return_hydra_config=True,
        )

    providers = [instantiate(provider) for provider in config.dataset.resource_providers]

    assert [provider.name for provider in providers] == ["transcript", "spkid"]
    assert set(config.columns) == {"utt_id", "path", "split", "transcript", "spkid"}


def test_vq_asr_config_exposes_an_inference_ready_system():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        config = compose(
            config_name="run/train_vq_asr_librispeech",
            return_hydra_config=True,
        )

    system = config.system
    assert "architecture" not in config
    assert system._target_ == "quick_convert.systems.asr.VQASRSystem"
    assert config.trainer.module.system == system
    assert config.trainer._target_ == "quick_convert.training.lightning.trainer.LightningTrainer"
    assert config.trainer.module._target_ == "quick_convert.training.lightning.modules.vq_asr.VQASRTrainingModule"
    assert not {"project_name", "exp_name", "batch_size", "val_batch_size", "dataloader"}.intersection(config.pipeline)
    assert config.trainer.train_dataloader_kwargs.batch_size == 32
    assert config.trainer.val_dataloader_kwargs.batch_size == 32
    assert "quantizer" not in config.trainer.module
    assert "ctc_head" not in config.trainer.module
    from quick_convert.components.ssl import W2VBertContentEncoder

    assert config.system.quantizer.input_dim == W2VBertContentEncoder.FEATURE_DIM


@pytest.mark.parametrize(
    ("config_path", "target"),
    [
        ("components/ssl/w2vbert.yaml", "quick_convert.components.ssl.W2VBertContentEncoder"),
        ("components/ssl/emo2vec.yaml", "quick_convert.components.ssl.EmotionEncoder"),
        ("components/ssl/dac.yaml", "quick_convert.components.ssl.DACContentEncoder"),
        ("components/ssl/pros2vec.yaml", "quick_convert.components.ssl.ProsodyEncoder"),
        ("components/speaker_encoder/espnet.yaml", "quick_convert.components.speaker.ESPnetSpeakerEncoder"),
        (
            "components/speaker_encoder/pyannote_wespeaker_voxceleb_resnet34_LM.yaml",
            "quick_convert.components.speaker.PyannoteWeSpeakerEncoder",
        ),
    ],
)
def test_public_component_configs_only_pass_declared_arguments(config_path, target):
    from hydra.utils import get_class

    config = OmegaConf.load(CONFIG_DIR / config_path)
    signature = inspect.signature(get_class(target))

    assert not any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
    assert {key for key in config if not key.startswith("_")} <= set(signature.parameters)


@pytest.mark.parametrize(
    ("config_path", "target"),
    [
        ("components/ssl/emo2vec.yaml", "quick_convert.components.ssl.EmotionEncoder"),
        ("components/ssl/dac.yaml", "quick_convert.components.ssl.DACContentEncoder"),
        ("components/ssl/pros2vec.yaml", "quick_convert.components.ssl.ProsodyEncoder"),
        ("components/speaker_encoder/espnet.yaml", "quick_convert.components.speaker.ESPnetSpeakerEncoder"),
    ],
)
def test_composition_sample_rate_matches_constructor_default(config_path, target):
    from hydra.utils import get_class

    config = OmegaConf.load(CONFIG_DIR / config_path)
    constructor_default = inspect.signature(get_class(target)).parameters["sample_rate"].default

    assert config.sample_rate == constructor_default


def test_w2vbert_exposes_sample_rate_as_a_fixed_capability():
    from quick_convert.components.ssl import W2VBertContentEncoder

    signature = inspect.signature(W2VBertContentEncoder)

    assert "sample_rate" not in signature.parameters
    assert W2VBertContentEncoder.SAMPLE_RATE == 16_000


def test_w2vbert_precompute_rate_matches_encoder_capability():
    from quick_convert.components.ssl import W2VBertContentEncoder

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        config = compose(
            config_name="run/precompute_content_w2vbert_librispeech",
            return_hydra_config=True,
        )

    assert config.dataset.target_sr == W2VBertContentEncoder.SAMPLE_RATE


@pytest.mark.parametrize(
    "target",
    [
        "quick_convert.components.decoders.flow_matching.estimators.residual_mlp.ResidualMLPEstimator",
        "quick_convert.components.decoders.flow_matching.base.BASECFM",
        "quick_convert.components.speaker.speaker_generators.cfm_speaker_generator.CFMSpeakerGenerator",
    ],
)
def test_configured_component_targets_are_importable(target):
    from hydra.utils import get_class

    assert get_class(target) is not None


def test_w2vbert_precompute_pipeline_instantiates_without_downloading_model(
    monkeypatch,
    tmp_path,
):
    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def to(self, device):
            return self

        def eval(self):
            return self

    fake_transformers = SimpleNamespace(
        AutoFeatureExtractor=FakeProcessor,
        AutoModel=FakeModel,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    libri_root = tmp_path / "librispeech" / "Librispeech"
    for split in (
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
    ):
        (libri_root / split).mkdir(parents=True)

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        config = compose(
            config_name="run/precompute_content_w2vbert_librispeech",
            overrides=[
                f"data_root={tmp_path}",
                f"out_root={tmp_path / 'out'}",
                f"pipeline.out_dir={tmp_path / 'features'}",
                "device=cpu",
            ],
        )

    pipeline = instantiate(config.pipeline)

    assert pipeline.extractor.encoder.model_name == "facebook/w2v-bert-2.0"
    assert pipeline.dataset.target_sr == 16_000


def test_clac_uses_base_dataset_with_split_qualified_ids(monkeypatch, tmp_path):
    monkeypatch.setenv("QUICK_CONVERT_CLAC_ROOT", str(tmp_path))

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        config = compose(config_name="dataset/clac")

    for split in config.dataset.splits:
        (tmp_path / split).mkdir()
    (tmp_path / "picnic" / "1234.wav").touch()

    dataset = instantiate(config.dataset)

    assert config.dataset.root == str(tmp_path)
    assert config.dataset._target_ == "quick_convert.data.BaseDataset"
    assert config.dataset.utt_id_template == "{path.parent.stem}/{path.stem}"
    assert [row.utt_id for row in dataset.rows] == ["picnic/1234"]


@pytest.mark.parametrize(
    "config_name",
    [
        "run/train_sslr_w2vbert_cmdiff",
        "run/train_sslr_w2vbert_dit",
    ],
)
def test_ssl_reconstruction_config_exposes_an_inference_ready_system(config_name):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        config = compose(
            config_name=config_name,
            return_hydra_config=True,
        )

    system = config.system
    assert "architecture" not in config
    assert system._target_ == "quick_convert.systems.reconstruction.SSLReconstructionSystem"
    assert config.trainer.module.system == system
    assert config.trainer._target_ == "quick_convert.training.lightning.trainer.LightningTrainer"
    assert (
        config.trainer.module._target_
        == "quick_convert.training.lightning.modules.ssl_reconstruction.SSLReconstructionTrainingModule"
    )
    assert "encoder" not in system
    assert "decoder" not in config.trainer.module
    assert "feature_transform" not in config.trainer.module
    assert "online_encoders" not in config.trainer.module


def test_rvq_ssl_reconstruction_uses_plain_quantizer_encoder():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        config = compose(
            config_name="run/train_sslr_w2vbert_cmdiff_rvq",
            return_hydra_config=True,
        )

    system = config.system
    encoder = system.encoder
    assert system._target_ == "quick_convert.systems.reconstruction.SSLReconstructionSystem"
    assert config.trainer.module.system == system
    assert encoder._target_ == "quick_convert.components.layers.rvq_ema.ResidualVectorQuantizerEMA"
    assert encoder.input_dim == 1024
    assert system.decoder.feature_dim == encoder.input_dim
    assert "encoder" not in config.trainer.module

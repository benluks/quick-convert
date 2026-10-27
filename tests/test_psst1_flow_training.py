from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from quick_convert.components.speaker.speaker_generators.psst1_flow import PSST1FlowSpeakerGenerator
from quick_convert.data import BaseDataset
from quick_convert.data.resources import PathResourceProvider
from quick_convert.utils.resolvers import class_attribute


CONFIG_DIR = Path(__file__).parents[1] / "quick_convert" / "configs"


def test_fit_standardization_uses_population_statistics():
    generator = PSST1FlowSpeakerGenerator()
    embeddings = torch.randn(64, 192) * 3 + 5
    embeddings[:, 0] = 1.0

    generator.fit_standardization(embeddings)

    assert torch.allclose(generator.mean, embeddings.mean(0), atol=1e-5)
    assert torch.allclose(generator.std[1:], embeddings[:, 1:].std(0, correction=0), atol=1e-5)
    assert generator.std[0] == 1.0
    standardized = generator.standardize(embeddings)
    assert torch.allclose(standardized[:, 1:].mean(0), torch.zeros(191), atol=1e-4)


def test_saved_generator_round_trips(tmp_path):
    generator = PSST1FlowSpeakerGenerator()
    generator.fit_standardization(torch.randn(16, 192))

    generator.save(tmp_path / "flow.pt")
    restored = PSST1FlowSpeakerGenerator(tmp_path / "flow.pt")

    z = torch.randn(2, 192)
    assert torch.equal(restored.mean, generator.mean)
    assert torch.equal(restored.std, generator.std)
    assert torch.allclose(restored.generate(z, 3), generator.eval().generate(z, 3))


def test_flow_loss_is_differentiable():
    generator = PSST1FlowSpeakerGenerator()

    loss = generator.compute_loss(torch.randn(8, 192))["loss"]
    loss.backward()

    assert torch.isfinite(loss)
    assert generator.velocity_net.out_proj.weight.grad is not None


@pytest.fixture
def embedding_dataset(tmp_path):
    """A tiny LibriSpeech-shaped split with one precomputed embedding per file."""
    generator = torch.Generator().manual_seed(0)
    audio_root, embeddings_root = tmp_path / "audio", tmp_path / "ecapa"
    for index in range(20):
        audio = audio_root / "train-clean-360" / "1" / "2" / f"1-2-{index:04d}.flac"
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.touch()
        target = embeddings_root / "train-clean-360" / f"{audio.stem}.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.randn(192, generator=generator) * 10 + 2, target)

    return BaseDataset(
        root=audio_root,
        splits=["train-clean-360"],
        file_format="flac",
        utt_id_template="{path.stem}",
        resource_providers=[
            PathResourceProvider(
                name="speaker_embedding",
                kind="torch_tensor",
                path_template=str(embeddings_root) + "/{sample.split}/{sample.utt_id}.pt",
            )
        ],
        load=["speaker_embedding"],
    )


def compose_module(tmp_path):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        config = compose(
            config_name="run/train_psst1_flow_librispeech",
            overrides=[f"trainer.module.export_path={tmp_path}/flow.pt"],
        )
    return instantiate(config.trainer.module), tmp_path / "flow.pt"


def test_training_module_fits_statistics_and_exports_flow(tmp_path, embedding_dataset):
    pytest.importorskip("lightning")
    import lightning as L

    module, export_path = compose_module(tmp_path)
    module.setup_training(embedding_dataset)
    expected = torch.stack([sample.resources["speaker_embedding"].value for sample in embedding_dataset])

    assert torch.allclose(module.generator.mean, expected.mean(0), atol=1e-4)
    assert torch.allclose(module.generator.std, expected.std(0, correction=0), atol=1e-4)

    trainer = L.Trainer(
        max_steps=3,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(module, embedding_dataset.make_dataloader(batch_size=8, shuffle=True))

    restored = PSST1FlowSpeakerGenerator(export_path)
    assert torch.equal(restored.mean, module.generator.mean.cpu())
    assert all(
        torch.equal(value, module.generator.velocity_net.state_dict()[key].cpu())
        for key, value in restored.velocity_net.state_dict().items()
    )


def test_training_run_reads_precomputed_ecapa_embeddings():
    if not OmegaConf.has_resolver("class_attr"):
        OmegaConf.register_new_resolver("class_attr", class_attribute)
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR.resolve())):
        train = compose(config_name="run/train_psst1_flow_librispeech", return_hydra_config=True)
        precompute = compose(config_name="run/precompute_spk_ecapa_librispeech", return_hydra_config=True)

    assert list(train.train_dataset.splits) == list(precompute.dataset.splits) == ["train-clean-360"]
    assert train.embeddings_path == "outputs/precomputed/librispeech/ecapa"
    assert precompute.hydra.runtime.choices.dataset == "librispeech"
    assert OmegaConf.to_container(precompute.pipeline)["out_dir"].endswith(
        "/precomputed/${hydra:runtime.choices.dataset}/ecapa"
    )
    assert OmegaConf.to_container(train.trainer.module)["export_path"] == "${pipeline.out_dir}/flow.pt"
    assert train.trainer.trainer_kwargs.max_epochs == 301
    assert train.trainer.train_dataloader_kwargs.batch_size == 128
    assert train.trainer.module.seed == 1999
    assert train.trainer.trainer_kwargs.logger is True  # Lightning default: local TensorBoard

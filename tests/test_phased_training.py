from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader


pytest.importorskip("lightning")

import lightning as L

from quick_convert.training.lightning.modules.base import BaseTrainingModule
from quick_convert.training.lightning.optim import Optimization


@dataclass
class Output:
    loss: torch.Tensor


class ToyGAN(BaseTrainingModule):
    def __init__(self, optimization=None):
        super().__init__(
            optimization=optimization
            if optimization is not None
            else {
                name: Optimization(
                    optimizer=torch.optim.SGD,
                    optimizer_kwargs={"lr": 0.1},
                    lr_scheduler=torch.optim.lr_scheduler.ExponentialLR,
                    lr_scheduler_kwargs={"gamma": 0.5},
                    interval="epoch",
                )
                for name in ("discriminator", "generator")
            }
        )
        self.generator = torch.nn.Linear(1, 1, bias=False)
        self.discriminator = torch.nn.Linear(1, 1, bias=False)
        self.generator.weight.data.fill_(0.5)
        self.discriminator.weight.data.fill_(0.5)
        self.events = []

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch.to(device)

    def phase_parameters(self):
        return {
            "generator": self.generator.parameters(),
            "discriminator": self.discriminator.parameters(),
        }

    def phase_step(self, batch, phase):
        self.events.append(phase)
        fake = self.generator(batch)
        if phase == "discriminator":
            assert not self.generator.weight.requires_grad
            return Output(self.discriminator(fake.detach()).square().mean())
        assert not self.discriminator.weight.requires_grad
        return Output((self.discriminator(fake) - 1).square().mean())

    def _shared_step(self, batch, stage):
        return Output(self.generator(batch).square().mean())


def test_phases_update_and_schedule():
    model = ToyGAN()
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, train_dataloaders=DataLoader(torch.ones(2, 1), batch_size=1))
    assert model.events == ["discriminator", "generator"] * 2
    assert trainer.global_step == 4
    assert model.generator.weight.item() > 0.5
    assert model.discriminator.weight.item() < 0.5
    assert all(opt.param_groups[0]["lr"] == pytest.approx(0.05) for opt in trainer.optimizers)
    assert all(p.requires_grad for p in model.parameters())


def test_overlapping_groups_rejected():
    model = ToyGAN()
    model.phase_parameters = lambda: {name: model.parameters() for name in model.phase_names}
    with pytest.raises(ValueError, match="overlapping"):
        model.configure_optimizers()


def test_hydra_gan_recipe_instantiates_phased_optimizers():
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    configs = Path(__file__).resolve().parents[1] / "quick_convert/configs"
    recipe = OmegaConf.load(configs / "run/train_waveform_knnvc_librispeech.yaml")
    module = recipe.trainer.module
    config = OmegaConf.create(
        {
            "_target_": f"{ToyGAN.__module__}.ToyGAN",
            "_convert_": module.get("_convert_", "none"),
            "optimization": OmegaConf.load(configs / "optimization/gan.yaml"),
        }
    )
    model = instantiate(config)
    assert model.phase_names == ("discriminator", "generator")
    assert all(isinstance(value, Optimization) for value in model.optimization.values())
    optimizers, schedulers = model.configure_optimizers()
    assert len(optimizers) == len(schedulers) == 2
    for optimizer, scheduler in zip(optimizers, schedulers, strict=True):
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0002)
        assert isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)
        assert scheduler.gamma == pytest.approx(0.999)

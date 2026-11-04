"""Exercise Lightning's real Hydra child-process relaunch through the console script."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_console_entry_point_launches_two_cpu_ddp_ranks(tmp_path):
    pytest.importorskip("lightning")
    pytest.importorskip("wandb")
    executable = Path(sys.executable).parent / "quick-convert"
    if not executable.is_file():
        pytest.skip("Installed console entry point is required")
    (tmp_path / "ddp_smoke.py").write_text("""
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from quick_convert.training.lightning.trainer import LightningTrainer
from quick_convert.pipelines.training.pipeline import TrainingPipeline

class Model(LightningModule):
    def __init__(self, output_dir):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)
        self.output_dir = output_dir
    def setup_training(self, dataset):
        pass
    def training_step(self, batch, batch_idx):
        return self.layer(batch[0]).square().mean()
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)
    def on_train_end(self):
        Path(self.output_dir, f'rank-{self.global_rank}').write_text(str(self.global_step))

class Dataset:
    def make_dataloader(self, **kwargs):
        return DataLoader(TensorDataset(torch.ones(8, 2)), batch_size=2)

class Pipeline(TrainingPipeline):
    def __init__(self, output_dir):
        trainer = LightningTrainer(Model(output_dir), trainer_kwargs=dict(
            accelerator='cpu', devices=2, strategy='ddp', max_steps=2,
            logger=WandbLogger(project='ddp-smoke', save_dir=output_dir, offline=True),
            callbacks=[ModelCheckpoint(every_n_train_steps=1, save_last=True)],
            enable_progress_bar=False, enable_model_summary=False))
        super().__init__(trainer, Dataset(), out_dir=output_dir)
""")
    (tmp_path / "smoke.yaml").write_text(
        "pipeline:\n  _target_: ddp_smoke.Pipeline\n  output_dir: " + str(tmp_path) + "\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path), str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH", "")]
    )
    env["OMP_NUM_THREADS"] = "1"
    for key in ("LOCAL_RANK", "RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    result = subprocess.run(
        [
            str(executable),
            "train_vq_asr_librispeech",
            "--config-path",
            str(tmp_path),
            "--config-name",
            "smoke",
            "hydra.run.dir=" + str(tmp_path / "hydra"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "rank-0").read_text() == "2"
    assert (tmp_path / "rank-1").read_text() == "2"
    configs = list((tmp_path / "ddp-smoke").glob("*/config.yaml"))
    assert len(configs) == 1
    assert configs[0].parent.parent.name == "ddp-smoke"
    assert not (tmp_path / "config.yaml").exists()
    assert len(list((tmp_path / "wandb").glob("offline-run-*"))) == 1
    import torch

    saved = torch.load(configs[0].parent / "checkpoints" / "last.ckpt", weights_only=False)
    assert saved["quick_convert_logger_runs"][0]["id"] == configs[0].parent.name

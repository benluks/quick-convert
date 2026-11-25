import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from quick_convert.data.types import AudioBatch, AudioSample


def test_lightning_training_throughput(tmp_path):
    lightning = pytest.importorskip("lightning.pytorch")
    from quick_convert.training.lightning.throughput import TrainingThroughput

    class Model(lightning.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def training_step(self, batch, batch_idx):
            return self.weight.square()

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    samples = [
        AudioSample(utt_id=str(i), path=Path(str(i)), waveform=torch.ones(10), sample_rate=10)
        for i in range(8)
    ]
    callback = TrainingThroughput(warmup_batches=1)
    trainer = lightning.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
        callbacks=[callback],
    )
    trainer.fit(Model(), train_dataloaders=DataLoader(samples, batch_size=2, collate_fn=AudioBatch.from_samples))
    reports = json.loads((tmp_path / "throughput.json").read_text())
    assert [row["batches"] for row in reports] == [3, 4]
    assert [row["audio_seconds"] for row in reports] == [6, 8]
    assert [row["examples"] for row in reports] == [6, 8]
    assert all(row["audio_seconds_per_second"] > 0 for row in reports)
    assert all("peak_allocated_bytes" not in row for row in reports)

"""Standalone CPU DDP integration worker, launched by pytest."""

import json
import sys
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, DistributedSampler

from quick_convert.data.sampling import DistributedBatchSampler, LengthAwareBatchSampler
from quick_convert.training.lightning.trainer import LightningTrainer

LENGTHS = list(range(1, 26))


class Policy:
    def __init__(self, budget):
        self.budget = budget

    def build(self, dataset, *, batch_size, shuffle, drop_last):
        return LengthAwareBatchSampler(
            LENGTHS,
            batch_size,
            pool_size=8,
            seed=42,
            shuffle=shuffle,
            drop_last=drop_last,
            max_batch_cost=self.budget,
        )


class Dataset:
    def __len__(self):
        return len(LENGTHS)

    def make_dataloader(self, batch_size=4, shuffle=False, sampling=None, distributed=None):
        data = list(range(len(self)))
        rank, world_size = distributed
        if sampling is not None:
            sampler = sampling.build(self, batch_size=batch_size, shuffle=shuffle, drop_last=False)
            return DataLoader(data, batch_sampler=DistributedBatchSampler(sampler, rank=rank, world_size=world_size))
        sampler = DistributedSampler(data, num_replicas=world_size, rank=rank, shuffle=shuffle)
        return DataLoader(data, batch_size=batch_size, sampler=sampler)


class Model(L.LightningModule):
    def __init__(self, output):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.output = output
        self.train_seen = {}
        self.val_seen = {}

    def setup_training(self, dataset):
        pass

    def training_step(self, batch, batch_idx):
        self.train_seen.setdefault(str(self.current_epoch), []).append(batch.tolist())
        return self.weight.square()

    def validation_step(self, batch, batch_idx):
        self.val_seen.setdefault(str(self.current_epoch), []).extend(batch.tolist())

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def on_train_end(self):
        path = self.output / f"rank-{self.global_rank}.json"
        path.write_text(json.dumps({"train": self.train_seen, "val": self.val_seen}))


def main():
    output = Path(sys.argv[1])
    budget = None if sys.argv[2] == "fixed" else 60
    dataset = Dataset()
    policy = Policy(budget)
    trainer = LightningTrainer(
        Model(output),
        train_dataloader_kwargs={"sampling": policy, "batch_size": 4, "shuffle": True},
        val_dataloader_kwargs={"batch_size": 4},
        trainer_kwargs={
            "accelerator": "cpu",
            "devices": 2,
            "strategy": "ddp",
            "max_epochs": 3,
            "logger": False,
            "enable_checkpointing": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "num_sanity_val_steps": 0,
        },
    )
    trainer.prepare(dataset, output)
    trainer.train(dataset, dataset)
    if trainer.pl_trainer.global_rank != 0:
        return
    results = [json.loads((output / f"rank-{rank}.json").read_text()) for rank in range(2)]
    for epoch in range(3):
        sampler = policy.build(dataset, batch_size=4, shuffle=True, drop_last=False)
        sampler.set_epoch(epoch)
        for rank, result in enumerate(results):
            expected = list(DistributedBatchSampler(sampler, rank=rank, world_size=2))
            assert result["train"][str(epoch)] == expected
            val = DistributedSampler(list(range(len(dataset))), num_replicas=2, rank=rank, shuffle=False)
            assert result["val"][str(epoch)] == list(val)


if __name__ == "__main__":
    main()

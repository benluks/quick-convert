from types import SimpleNamespace

import pytest

from quick_convert.data.durations import resolve_durations
from quick_convert.data.sampling import LengthAwareBatchSampler, batching_statistics


def test_fixed_batches():
    lengths = list(range(1, 101))
    sampler = LengthAwareBatchSampler(lengths, 8, seed=42)
    batches = list(sampler)
    assert sorted(i for batch in batches for i in batch) == list(range(100))
    assert len(batches) == len(sampler)
    assert batches == list(sampler)
    sampler.set_epoch(1)
    assert batches != list(sampler)
    baseline = [list(range(i, min(i + 8, 100))) for i in range(0, 100, 8)]
    assert batching_statistics(lengths, batches)["padding_fraction"] < 0.1
    assert batching_statistics(lengths, baseline)["examples"] == 100


@pytest.mark.parametrize("power", [1, 2])
def test_dynamic_budget(power):
    lengths = list(range(1, 21))
    sampler = LengthAwareBatchSampler(lengths, 8, max_batch_cost=400, cost_power=power)
    batches = list(sampler)
    assert sorted(i for batch in batches for i in batch) == list(range(20))
    assert all(len(b) <= 8 and len(b) * max(lengths[i] for i in b) ** power <= 400 for b in batches)


def test_duration_cache(tmp_path):
    path = tmp_path / "audio.flac"
    path.touch()
    rows = [
        SimpleNamespace(path=path, start=None, end=None, utt_id="a"),
        SimpleNamespace(path=path, start=2, end=5, utt_id="b"),
    ]
    calls = []

    def reader(p):
        calls.append(p)
        return 10

    cache = tmp_path / "durations.json"
    assert resolve_durations(rows, cache_path=cache, duration_reader=reader) == [10, 3]
    assert resolve_durations(rows, cache_path=cache, duration_reader=reader) == [10, 3]
    assert len(calls) == 1


def test_lightning_advances_batch_sampler_epoch():
    lightning = pytest.importorskip("lightning.pytorch")
    import torch
    from torch.utils.data import DataLoader

    lengths = list(range(1, 65))
    sampler = LengthAwareBatchSampler(lengths, 4, pool_size=16, seed=42)

    class Model(lightning.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.seen = {}

        def training_step(self, batch, batch_idx):
            self.seen.setdefault(self.current_epoch, []).append(batch.tolist())
            return self.weight.square()

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.01)

    model = Model()
    trainer = lightning.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=3,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        use_distributed_sampler=False,
    )
    trainer.fit(model, train_dataloaders=DataLoader(list(range(64)), batch_sampler=sampler))
    for epoch in range(3):
        sampler.set_epoch(epoch)
        assert model.seen[epoch] == list(sampler)
    assert model.seen[0] != model.seen[1]
    assert model.seen[1] != model.seen[2]

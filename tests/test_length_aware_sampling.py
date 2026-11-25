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


def test_composable_sampling_policy(tmp_path):
    import soundfile as sf
    import torch
    from hydra.utils import instantiate

    from quick_convert.data import BaseDataset

    paths = []
    for i in range(4):
        path = tmp_path / f"{i}.wav"
        sf.write(path, torch.zeros(1600 * (i + 1)).numpy(), 16000)
        paths.append(path)
    dataset = BaseDataset(paths=paths, utt_id_template="{path.stem}")
    policy = instantiate(
        {
            "_target_": "quick_convert.data.sampling.LengthAwareSampling",
            "pool_size": 4,
            "max_batch_seconds": 1.0,
            "max_batch_attention_cost": 0.2,
        }
    )
    loader = dataset.make_dataloader(batch_size=2, sampling=policy)
    assert isinstance(loader.batch_sampler, LengthAwareBatchSampler)
    assert sorted(i for batch in loader.batch_sampler for i in batch) == list(range(4))
    assert all(len(batch) <= 2 for batch in loader.batch_sampler)
    assert loader.batch_sampler.budgets == {1: 1.0, 2: 0.2}
    with pytest.raises(ValueError, match="another sampler"):
        dataset.make_dataloader(sampling=policy, batch_sampler=loader.batch_sampler)


def test_dataset_accepts_independent_sampling_policy():
    from quick_convert.data import BaseDataset

    class ReverseSampling:
        def build(self, dataset, *, batch_size, shuffle, drop_last):
            assert batch_size == 2
            assert not shuffle
            assert not drop_last
            return [[1, 0]]

    dataset = BaseDataset(rows=[])
    loader = dataset.make_dataloader(batch_size=2, sampling=ReverseSampling())
    assert list(loader.batch_sampler) == [[1, 0]]


@pytest.mark.parametrize("extension", ["wav", "flac"])
def test_duration_metadata_without_ffprobe(tmp_path, monkeypatch, extension):
    import soundfile as sf
    import torch

    from quick_convert.data import durations

    def unexpected_probe(path):
        pytest.fail(f"Unexpected ffprobe call for {path}")

    monkeypatch.setattr(durations, "probe_audio_duration", unexpected_probe)
    path = tmp_path / f"audio.{extension}"
    sf.write(path, torch.zeros(4000).numpy(), 16000)
    assert durations.read_audio_duration(path) == pytest.approx(0.25)


def test_duration_metadata_fallback(tmp_path, monkeypatch):
    from quick_convert.data import durations

    path = tmp_path / "audio.mp4"
    path.touch()
    monkeypatch.setattr(durations, "probe_audio_duration", lambda p: 2.5)
    assert durations.read_audio_duration(path) == 2.5


@pytest.mark.parametrize("drop_remainder", [False, True])
def test_distributed_batch_schedule(drop_remainder):
    from quick_convert.data.sampling import DistributedBatchSampler

    global_sampler = LengthAwareBatchSampler(list(range(1, 26)), 4, pool_size=8, seed=42)
    wrappers = [
        DistributedBatchSampler(global_sampler, rank=rank, world_size=3, drop_remainder=drop_remainder)
        for rank in range(3)
    ]
    for epoch in range(3):
        for wrapper in wrappers:
            wrapper.set_epoch(epoch)
        expected = list(global_sampler)
        actual = [list(wrapper) for wrapper in wrappers]
        assert all(len(batches) == len(wrappers[0]) for batches in actual)
        reconstructed = [batch for step in zip(*actual, strict=True) for batch in step]
        if drop_remainder:
            assert reconstructed == expected[:6]
        else:
            assert reconstructed == expected + expected[:2]


def test_lightning_distributed_loader_construction():
    pytest.importorskip("lightning.pytorch")
    from types import SimpleNamespace

    from quick_convert.data import BaseDataset
    from quick_convert.data.sampling import DistributedBatchSampler
    from quick_convert.training.lightning.trainer import _SamplingDataModule

    class Policy:
        def build(self, dataset, *, batch_size, shuffle, drop_last):
            return LengthAwareBatchSampler([1] * 8, batch_size, shuffle=shuffle, drop_last=drop_last)

    dataset = BaseDataset(rows=[])
    datamodule = _SamplingDataModule(dataset, dataset, {"sampling": Policy(), "batch_size": 2}, {})
    datamodule.trainer = SimpleNamespace(world_size=2, global_rank=1)
    loader = datamodule.train_dataloader()
    assert isinstance(loader.batch_sampler, DistributedBatchSampler)
    assert list(loader.batch_sampler) == [[2, 3], [6, 7]]
    assert datamodule.val_dataloader().sampler.rank == 1


@pytest.mark.parametrize("budget", ["fixed", "dynamic"])
def test_lightning_multiprocess_ddp(tmp_path, budget):
    pytest.importorskip("lightning.pytorch")
    import os
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).with_name("sampling_ddp_worker.py")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, str(script), str(tmp_path), budget],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_hybrid_budget():
    lengths = list(range(1, 21)) * 8
    sampler = LengthAwareBatchSampler(
        lengths, 8, pool_size=32, max_batch_seconds=64, max_batch_attention_cost=400
    )
    for epoch in range(3):
        sampler.set_epoch(epoch)
        batches = list(sampler)
        assert sorted(i for batch in batches for i in batch) == list(range(len(lengths)))
        assert len(batches) == len(sampler)
        for batch in batches:
            longest = max(lengths[i] for i in batch)
            assert len(batch) <= 8
            assert len(batch) * longest <= 64
            assert len(batch) * longest**2 <= 400


@pytest.mark.parametrize(
    "options",
    [
        {"max_batch_seconds": 0},
        {"max_batch_attention_cost": 399},
        {"max_batch_seconds": 64, "max_batch_cost": 64},
        {"max_batch_attention_cost": 400, "drop_last": True},
    ],
)
def test_invalid_hybrid_budget(options):
    with pytest.raises(ValueError):
        LengthAwareBatchSampler([20], 8, **options)

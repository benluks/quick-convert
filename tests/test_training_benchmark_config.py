from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir


@pytest.mark.parametrize(
    ("name", "batch_size", "linear", "quadratic"),
    [
        ("random", 32, None, None),
        ("fixed", 32, None, None),
        ("linear", 64, 512, None),
        ("quadratic", 64, None, 8000),
        ("hybrid", 64, 512, 8000),
    ],
)
def test_training_benchmark_config(name, batch_size, linear, quadratic):
    config_dir = Path(__file__).resolve().parents[1] / "quick_convert" / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="run/benchmark_training_librispeech",
            overrides=[f"sampling@trainer.train_dataloader_kwargs={name}"],
        )
    options = cfg.trainer.train_dataloader_kwargs
    assert options.batch_size == batch_size
    assert options.drop_last is False
    assert cfg.val_dataset is None
    assert cfg.trainer.trainer_kwargs.devices == 1
    assert cfg.trainer.trainer_kwargs.logger is False
    if name == "random":
        assert options.sampling is None
    else:
        assert options.sampling.pool_size == 1024
        assert options.sampling.get("max_batch_seconds") == linear
        assert options.sampling.get("max_batch_attention_cost") == quadratic

from types import SimpleNamespace

import pytest
import torch


pytest.importorskip("lightning")

from quick_convert.training.lightning.logging.media_logger import TensorBoardMediaLogger


def test_contour_renders_masked_targets_without_mutation():
    from types import SimpleNamespace

    pytest.importorskip("lightning")
    from quick_convert.training.lightning.logging.media_logger import MediaLogger

    images = {}
    module = SimpleNamespace(
        global_step=10,
        log_images=lambda values, **kwargs: images.update(values),
    )
    from quick_convert.components.heads.base import HeadTarget

    target = HeadTarget(
        torch.tensor([[0, 999, 2]]),
        observation_mask=torch.tensor([[True, False, True]]),
        timestamps=torch.tensor([[0.0, 0.02, 0.04]]),
    )
    MediaLogger.log_frame_predictions(
        module,
        "val/prosody/pitch/contour",
        torch.tensor([[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]]),
        target,
        torch.ones(1, 3, dtype=torch.bool),
        kind="bins",
        step=10,
    )
    image = images["val/prosody/pitch/contour"]
    assert image.shape[0] == 3
    assert image.shape[1] > 100 and image.shape[2] > 100
    assert torch.isfinite(image).all()
    assert target.values.tolist() == [[0, 999, 2]]


def test_wandb_bin_table_uses_one_native_table():
    from quick_convert.training.lightning.logging.media_logger import WandbMediaLogger

    wandb = pytest.importorskip("wandb")
    calls = []
    logger = WandbMediaLogger(
        SimpleNamespace(experiment=SimpleNamespace(log=lambda *args, **kwargs: calls.append((args, kwargs))))
    )
    columns = ["bin", "target_fraction", "predicted_fraction", "recall"]
    rows = [[0, 0.5, 1.0, 1.0], [1, 0.5, 0.0, None]]
    logger.log_table("val/prosody/pitch/bins", columns, rows, step=7)
    assert len(calls) == 1
    table = calls[0][0][0]["val/prosody/pitch/bins"]
    assert isinstance(table, wandb.Table)
    assert table.columns == columns
    assert table.data == rows
    assert calls[0][1] == {"step": 7}


def test_tensorboard_bin_table_uses_text():
    calls = []
    logger = TensorBoardMediaLogger(
        SimpleNamespace(experiment=SimpleNamespace(add_text=lambda *args, **kwargs: calls.append((args, kwargs))))
    )
    logger.log_table("bins", ["bin", "recall"], [[0, None]], step=7)
    assert calls == [(("bins", "bin | recall\n--- | ---\n0 | "), {"global_step": 7})]

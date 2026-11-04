import inspect
from types import SimpleNamespace

import pytest
import torch


pytest.importorskip("lightning")

from quick_convert.pipelines.training.logging.media_logger import (
    ReconstructedAudio as LegacyReconstructedAudio,
)
from quick_convert.training.lightning.logging.media_logger import (
    NullMediaLogger,
    ReconstructedAudio,
    TensorBoardMediaLogger,
    make_media_logger,
)


def test_legacy_media_logger_import_is_compatible():
    assert LegacyReconstructedAudio is ReconstructedAudio


def test_supported_media_loggers_are_concrete():
    assert not inspect.isabstract(TensorBoardMediaLogger)
    assert not inspect.isabstract(NullMediaLogger)


def test_tensorboard_logs_reconstructed_media_and_text():
    calls = []

    class RecordingExperiment:
        def add_audio(self, key, value, **kwargs):
            calls.append(("audio", key, tuple(value.shape), kwargs))

        def add_image(self, key, value, **kwargs):
            calls.append(("image", key, tuple(value.shape), kwargs))

        def add_text(self, key, value, **kwargs):
            calls.append(("text", key, value, kwargs))

    logger = TensorBoardMediaLogger(SimpleNamespace(experiment=RecordingExperiment()))
    media = ReconstructedAudio(
        reconstructed_audio=torch.zeros(1, 12),
        reconstructed_audio_lengths=torch.tensor([8]),
        reconstructed_mel=torch.zeros(1, 80, 5),
        reconstructed_mel_lengths=torch.tensor([3]),
        reconstructed_sample_rate=24_000,
        ids=["utterance"],
    )

    logger.log_reconstructed_audio("reconstruction", media, step=7)
    logger.log_text({"asr": "hello"}, step=7)

    assert calls == [
        (
            "audio",
            "reconstruction/utterance/reconstructed_audio",
            (1, 8),
            {"global_step": 7, "sample_rate": 24_000},
        ),
        (
            "image",
            "reconstruction/utterance/reconstructed_mel",
            (1, 80, 3),
            {"global_step": 7},
        ),
        ("text", "asr", "hello", {"global_step": 7}),
    ]


def test_disabled_logger_accepts_training_media():
    logger = make_media_logger(None)
    assert isinstance(logger, NullMediaLogger)
    logger.log_bar("layer_weights", torch.ones(24), "layer", "weight", step=0)


@pytest.mark.parametrize("shape", [(3, 300, 1000), (1, 20, 30), (20, 30)])
def test_wandb_image_adapter_uses_channel_last_without_mutation(shape):
    from quick_convert.training.lightning.logging.media_logger import WandbMediaLogger

    wandb = pytest.importorskip("wandb")
    calls = []
    logger = WandbMediaLogger(
        SimpleNamespace(experiment=SimpleNamespace(log=lambda *args, **kwargs: calls.append((args, kwargs))))
    )
    image = torch.zeros(shape)
    original = image.clone()
    logger.log_images({"contour": image}, step=7)
    media = calls[0][0][0]["contour"]
    assert isinstance(media, wandb.Image)
    assert media.image.size == (shape[-1], shape[-2])
    assert calls[0][1] == {"step": 7}
    torch.testing.assert_close(image, original)

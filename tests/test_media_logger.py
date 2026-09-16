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

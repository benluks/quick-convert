from __future__ import annotations

from typing import Any

import torch

from quick_convert.utils.audio import load_audio_input
from quick_convert.utils.device import DeviceLike, configure_device

from .base import SERSystem


class SpeechBrainSER(SERSystem):
    """Categorical speech-emotion recognition with a SpeechBrain interface.

    The default model is ``speechbrain/emotion-recognition-wav2vec2-IEMOCAP``,
    which predicts ``neu``, ``hap``, ``sad`` or ``ang``. Loading downloads the
    model and runs its ``pymodule_file`` from the Hugging Face Hub. The model is
    loaded on first use.

    Requires the ``speechbrain-ser`` extra.
    """

    sample_rate = 16_000

    def __init__(
        self,
        source: str = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        pymodule_file: str = "custom_interface.py",
        classname: str = "CustomEncoderWav2vec2Classifier",
        savedir: str | None = None,
        device: DeviceLike = None,
    ):
        super().__init__()
        self.source = source
        self.pymodule_file = pymodule_file
        self.classname = classname
        self.savedir = savedir
        self.device = configure_device(device)
        self._classifier = None

    def _get_classifier(self):
        if self._classifier is None:
            from speechbrain.inference.interfaces import foreign_class

            self._classifier = foreign_class(
                source=self.source,
                pymodule_file=self.pymodule_file,
                classname=self.classname,
                savedir=self.savedir,
                run_opts={"device": str(self.device)},
            )
        return self._classifier

    @torch.inference_mode()
    def predict(self, waveform: torch.Tensor, sample_rate: int) -> str:
        """Return the model's label for one waveform, lower-cased."""
        audio = load_audio_input(waveform, target_sample_rate=self.sample_rate, sample_rate=sample_rate)
        _, _, _, labels = self._get_classifier().classify_batch(audio.to(self.device))
        return str(labels[0]).strip().lower()

    def get_labels(self, batch: Any) -> list[str]:
        labels = []
        for sample in batch:
            if getattr(sample, "waveform", None) is not None:
                labels.append(self.predict(sample.waveform, int(sample.sample_rate)))
            else:
                labels.append(
                    self.predict(load_audio_input(sample.path, target_sample_rate=self.sample_rate), self.sample_rate)
                )
        return labels

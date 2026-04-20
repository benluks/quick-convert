# quick_convert/components/speaker_encoders/pyannote_wespeaker.py
from __future__ import annotations

from pathlib import Path

from pyannote.audio import Inference, Model

from .base import SpeakerEmbedding, SpeakerEncoder


class PyannoteWeSpeakerEncoder(SpeakerEncoder):
    def __init__(
        self,
        model_name: str = "pyannote/wespeaker-voxceleb-resnet34-LM",
        window: str = "whole",
    ) -> None:
        self.model_name = model_name
        self.model = Model.from_pretrained(model_name)
        self.inference = Inference(self.model, window=window)

    def encode_file(self, path: str | Path) -> SpeakerEmbedding:
        embedding = self.inference(str(path))  # numpy array, typically shape (1, D)

        if embedding.ndim == 2 and embedding.shape[0] == 1:
            embedding = embedding[0]

        return SpeakerEmbedding(
            values=embedding,
            backend="pyannote.audio",
            model_name=self.model_name,
            dim=int(embedding.shape[-1]),
        )

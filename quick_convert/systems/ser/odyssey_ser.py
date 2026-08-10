from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForAudioClassification

from .base import SERSystem


# Suppress warnings from transformers and torch
# warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
# warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
# logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


class OdysseySER(SERSystem):
    def __init__(
        self,
        device: str = "cpu",
        name: str = "OdysseySER",
        model_name: str = "3loi/SER-Odyssey-Baseline-WavLM-Categorical",
    ):
        """
        Initialize the OdysseySER metric and SER system.
        Uses lazy-loading for the SER model.
        """
        super().__init__()
        self.device = device if device != "mps" else "cpu"
        self.name = name
        self.model_name = model_name
        self._model = None
        self.mean = None
        self.std = None

    def _get_model(self):
        """
        Lazy-load the SER model and feature extractor.
        Also initialize normalization parameters (mean and std).
        """
        if self._model is None:
            self._model = AutoModelForAudioClassification.from_pretrained(self.model_name, trust_remote_code=True).to(
                self.device
            )
            self.mean = self._model.config.mean
            self.std = self._model.config.std
            self._model.eval()
        return self._model

    def get_labels(self, batch: Iterable[Any]) -> list:
        """
        For each sample in the batch, compute the SER reference embedding.
        This method iterates over the batch and returns a list of embeddings.
        """

        return [self.compute_embedding(sample) for sample in batch]

    def compute_embedding(self, sample: Any) -> np.ndarray:
        """
        Compute the SER embedding for a single sample by:
          1. Loading the audio file (from sample.path)
          2. Normalizing the audio using the model's mean and std.
          3. Forwarding the audio through the SER model to obtain logits,
             which are then converted to a probability distribution (embedding).
        """
        model = self._get_model()
        # Load audio using soundfile (assuming sample.path exists)
        audio, _ = sf.read(sample.path)
        # Normalize audio (model's mean and std are assumed to be scalars)
        norm_audio = (audio - self.mean) / self.std + 1e-9
        # Convert to tensor and add batch dimension
        audio_tensor = torch.tensor(norm_audio, dtype=torch.float32).unsqueeze(0).to(self.device)
        # Create a mask if required (matching the length of the audio)
        mask = torch.ones(1, audio_tensor.shape[1], device=self.device)
        with torch.no_grad():
            outputs = model(audio_tensor, mask)
        # Convert logits to probabilities (as our embedding)
        probabilities = torch.nn.functional.softmax(outputs, dim=-1)
        return probabilities.squeeze().cpu().numpy()

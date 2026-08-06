from __future__ import annotations

import numpy as np
import torch
import soundfile as sf
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from quick_convert.pipelines.evaluation.metrics.base import Metric
from quick_convert.systems.ser.odyssey_ser import OdysseySER
from typing import Any, Iterable

class MED(Metric):
    key = "ser_embedding"
    ref_key = "ref_ser_embedding"
    pred_key = "pred_ser_embedding"

    def __init__(self, 
                 device: str = "cpu", 
                 model_name: str = "3loi/SER-Odyssey-Baseline-WavLM-Categorical") -> None:
        """
        Initialize the SER metric by delegating embedding computation to OdysseySER.

        Args:
            device (str): The device to run the SER model on (e.g., 'cpu', 'cuda').
            model_name (str): The name of the model to load from Hugging Face.
        """
        self.device = device
        self.model_name = model_name
        self.odyssey_ser = OdysseySER(device=device, model_name=model_name)

    def get_references(self, batch: Iterable[Any]) -> dict[str, list]:
        """
        Use OdysseySER.get_labels() to compute the reference embeddings.

        Args:
            batch: Iterable containing the batch of audio samples.

        Returns:
            list: A list of reference embeddings.
        """
        return self.odyssey_ser.get_labels(batch)

    def compute(self, references: list, hypotheses: list) -> dict[str, float]:
        """
        Compute the Mean Euclidean Distance (MED) between the list of reference embeddings
        and hypothesis embeddings.

        Args:
            references (list): The list of reference embeddings.
            hypotheses (list): The list of hypothesis embeddings.

        Returns:
            dict[str, float]: The MED score.
        """
        if len(references) != len(hypotheses):
            raise ValueError("Number of references and hypotheses must be equal.")

        distances = []
        for ref, hyp in zip(references, hypotheses):
            distance = np.linalg.norm(np.array(ref) - np.array(hyp))
            distances.append(distance)

        med = float(np.mean(distances)) if distances else 0.0
        return {"ser_med": med}

from dataclasses import dataclass, replace
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from quick_convert.components.layers import AttentiveStatisticsPooling
from quick_convert.components.losses.speaker_losses import BaseSpeakerLoss


@dataclass
class SpeakerASRHeadOutput:
    speaker_features: torch.FloatTensor
    accuracy: Optional[torch.FloatTensor] = None
    predictions: Optional[torch.LongTensor] = None
    loss: Optional[torch.FloatTensor] = None


class SpeakerASPHead(nn.Module):
    """
    Simple speaker head that applies a linear layer to the content encoder output, followed by attentive statistics pooling and another linear layer.
    """

    def __init__(
        self,
        loss: BaseSpeakerLoss,
        input_dim: int = 512,
        hidden_dim: int = 128,
        output_dim: int = 192,
        supervision: str = "cosine",
        loss_index_key: str = "speaker",
    ):
        super().__init__()
        self.ln = nn.LayerNorm(input_dim)

        self.pre_pool = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.pool = AttentiveStatisticsPooling(input_dim=hidden_dim, hidden_dim=hidden_dim)
        self.post_pool = nn.Sequential(
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, output_dim),
        )

        self.output_dim = output_dim

        if supervision not in ["cosine", "aam"]:
            raise ValueError(f"Unsupported supervision type: {supervision}")
        self.supervision = supervision
        self.loss_index_key = loss_index_key

        if supervision == "aam":
            self.loss_partial = loss
        else:
            self.loss = loss

    def build_loss(self, indexers: dict[str, Any]):
        """
        A function for modules whose losses depend on a specific output shape.
        `indexers` is a dictionary of the dimenstion-bearing objects passed recursively
        through the root module. The key to access the correct object is defined in the constructor.
        You'll need to know how to determine the desired dimension from the object beforehand. This avoids needing
        to redundantly pass the loss dim in the hydra config (before the index is built).
        --
        This isn't ideal. Maybe in the future th ewprk should go towards pre-building the index in the first place
        so we know the number of speakers beforehand.
        """
        if self.supervision == "aam":
            num_speakers = len(indexers[self.loss_index_key])
            self.loss = self.loss_partial(
                num_classes=num_speakers,
            )

    def forward(self, x: torch.Tensor, padding_mask: torch.LongTensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, output_dim) output of the content encoder
        Returns:
            speaker_features: (B, output_dim)
        """
        x = self.ln(x)
        x = self.pre_pool(x)
        x = self.pool(x, padding_mask=padding_mask)
        x = self.post_pool(x)
        return SpeakerASRHeadOutput(speaker_features=x)

    def compute_loss(
        self,
        speaker_features: torch.FloatTensor,
        speaker_labels: torch.LongTensor = None,
        padding_mask: torch.LongTensor = None,
    ) -> SpeakerASRHeadOutput:
        """Compute cosine distance loss between predicted speaker features and target speaker embeddings."""

        spk_output = self.forward(speaker_features, padding_mask=padding_mask)
        x = spk_output.speaker_features
        # only need padding if
        if x.ndim == 3:
            raise NotImplementedError("Loss padding not yet implemented for frame-wise speaker embeddings")

        if self.supervision == "cosine":
            if speaker_labels is None:
                raise ValueError("Speaker embeddings must be provided for cosine supervision.")
            loss = self.loss(x, speaker_labels)
            accuracy = None
            preds = None

        elif self.supervision == "aam":
            if speaker_labels is None:
                raise ValueError("Speaker labels must be provided for AAM supervision.")
            # AAM loss has an internal classifier
            loss, accuracy, preds = self.loss(x, speaker_labels)

        else:
            raise ValueError(f"Unsupported supervision type: {self.supervision}")

        return replace(spk_output, loss=loss, accuracy=accuracy, predictions=preds)

from typing import Tuple

import torch
import torch.nn as nn

from quick_convert.components.layers import AttentiveStatisticsPooling

class SpeakerASPHead(nn.Module):
    """
    Simple speaker head that applies a linear layer to the content encoder output, followed by attentive statistics pooling and another linear layer.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
        output_dim: int = 192,
        supervision: str = "cosine",
    ):
        super().__init__()
        self.speaker_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            AttentiveStatisticsPooling(input_dim=hidden_dim, hidden_dim=hidden_dim),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, output_dim),
        )
        self.ln = nn.LayerNorm(input_dim)

        if supervision not in ["cosine", "aam"]:
            raise ValueError(f"Unsupported supervision type: {supervision}")
        self.supervision = supervision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, output_dim) output of the content encoder
        Returns:
            speaker_features: (B, output_dim)
        """
        x = self.ln(x)
        x = self.speaker_head(x)
        return x

    def compute_loss(
            self, 
            speaker_features: torch.FloatTensor, 
            speaker_embs: torch.FloatTensor = None, 
            speaker_labels: torch.LongTensor = None) -> Tuple:
        """Compute cosine distance loss between predicted speaker features and target speaker embeddings."""
        
        x = self.forward(speaker_features)
        # only need padding if
        if x.ndim == 3:
            raise NotImplementedError("Loss padding not yet implemented for frame-wise speaker embeddings")

        
        if self.supervision == "cosine":
            if speaker_embs is None:
                raise ValueError("Speaker embeddings must be provided for cosine supervision.")
            loss = self.loss(x, speaker_embs)
            accuracy = None
            preds = None

        elif self.supervision == "aam":
            if speaker_labels is None:
                raise ValueError("Speaker labels must be provided for AAM supervision.")
            # AAM loss has an internal classifier
            loss, accuracy, preds = self.loss(x, speaker_labels)

        else:
            raise ValueError(f"Unsupported supervision type: {self.supervision}")
        
        return x, loss, accuracy, preds

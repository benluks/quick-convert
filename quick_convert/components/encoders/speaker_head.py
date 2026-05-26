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
    ):
        super().__init__()
        self.speaker_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            AttentiveStatisticsPooling(input_dim=input_dim, hidden_dim=hidden_dim),
            nn.BatchNorm1d(input_dim * 2),
            nn.Linear(input_dim * 2, output_dim),
        )
        self.ln = nn.LayerNorm(input_dim)

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
    
    def compute_loss(self, speaker_features: torch.FloatTensor, speaker_embs: torch.FloatTensor) -> torch.Tensor:
        """ Compute cosine distance loss between predicted speaker features and target speaker embeddings. """
        x = self.forward(speaker_features)
        return 1 - torch.cosine_similarity(x, speaker_embs, dim=1).mean()

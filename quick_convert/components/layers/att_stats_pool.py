import torch
import torch.nn as nn

# From WeSpeaker: https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/models/pooling_layers.py

class AttentiveStatisticsPooling(nn.Module):
    """Attentive Statistics Pooling (compatible with WeSpeaker and
    W2V-BERT/WavLM)."""

    def __init__(
        self,
        in_planes=None,
        acoustic_dim=None,
        input_dim=None,
        hidden_dim=None,
    ):
        super().__init__()

        if input_dim is not None:
            # W2V / Transformer-style input
            self.feature_dim = input_dim
        elif in_planes is not None and acoustic_dim is not None:
            # WeSpeaker-style input
            outmap_size = int(acoustic_dim / 8)
            self.feature_dim = in_planes * 8 * outmap_size
        else:
            raise ValueError(
                "Specify either (in_planes, acoustic_dim) or "
                "(input_dim, hidden_dim)."
            )

        self.out_dim = self.feature_dim * 2
        hidden_dim = hidden_dim or 128

        self.attention = nn.Sequential(
            nn.Conv1d(self.feature_dim, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, self.feature_dim, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x: torch.Tensor):
        # W2V: [B, T, D]
        # WeSpeaker: [B, C, F, T]
        if x.dim() == 4:
            x = x.reshape(x.size(0), -1, x.size(-1))
        elif x.dim() == 3 and x.shape[1] != self.feature_dim:
            x = x.transpose(1, 2)

        w = self.attention(x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt(
            (
                torch.sum((x**2) * w, dim=2)
                - mu**2
            ).clamp(min=1e-5)
        )
        return torch.cat([mu, sg], dim=1)
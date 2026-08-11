from __future__ import annotations

import functools
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn

from quick_convert.components.layers import AttentiveStatisticsPooling
from quick_convert.components.layers.heads import HeadOutput, HeadTarget, SupervisedHead
from quick_convert.components.losses.speaker_losses import BaseSpeakerLoss


@dataclass
class SpeakerASRHeadOutput:
    speaker_features: torch.FloatTensor
    accuracy: torch.FloatTensor | None = None
    predictions: torch.LongTensor | None = None
    loss: torch.FloatTensor | None = None


class SpeakerASPHead(SupervisedHead):
    """
    Simple speaker head that applies a linear layer to the content encoder output, followed by attentive statistics pooling and another linear layer.
    """

    def __init__(
        self,
        loss: BaseSpeakerLoss,
        input_dim: int = 512,
        hidden_dim: int = 128,
        output_dim: int = 192,
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
        self.loss_index_key = loss_index_key
        self.loss = loss

    def build_loss(self, indexers: dict[str, Any]) -> None:
        """
        Build losses whose shape depends on an index created during dataset setup.

        For AAM supervision, the classifier output dimension is determined by
        the number of speakers in ``indexers[self.loss_index_key]``.
        """

        # build loss only needed for partial, because it depends on the training data (i.e. number of speakers)
        if not isinstance(self.loss, functools.partial):
            return

        if self.loss_index_key not in indexers:
            raise KeyError(
                f"Cannot build speaker AAM loss: no indexer named "
                f"{self.loss_index_key!r}. Available indexers: "
                f"{sorted(indexers)}"
            )

        num_speakers = len(indexers[self.loss_index_key])

        self.loss = self.loss(
            num_classes=num_speakers,
        )

    def forward(
        self,
        features: torch.Tensor,
        *,
        lengths: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> HeadOutput:
        """
        Produce one speaker embedding per sequence.

        Args:
            features:
                Routed frame-level features with shape ``(B, T, input_dim)``.
            lengths:
                Valid sequence lengths. Accepted for the generic head
                interface; currently the padding mask is used directly.
            padding_mask:
                Padding mask passed to attentive statistics pooling.

        Returns:
            A speaker-head output containing embeddings of shape
            ``(B, output_dim)``.
        """
        del lengths

        x = self.ln(features)
        x = self.pre_pool(x)
        x = self.pool(x, padding_mask=padding_mask)
        x = self.post_pool(x)

        return HeadOutput(features={"speaker_embedding": x})

    def predict(
        self,
        features: torch.Tensor,
        *,
        lengths: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> HeadOutput:
        return self.forward(
            features,
            lengths=lengths,
            padding_mask=padding_mask,
        )

    def compute_loss(
        self,
        features: torch.Tensor,
        *,
        targets: HeadTarget,
        lengths: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> HeadOutput:
        """Compute speaker-classification loss from pooled speaker embeddings."""

        spk_output = self.forward(features, padding_mask=padding_mask)
        embedding = spk_output.features["speaker_embedding"]
        # only need padding if
        if embedding.ndim == 3:
            raise NotImplementedError("Loss padding not yet implemented for frame-wise speaker embeddings")

        loss_output = self.loss(embedding, targets.values)

        return replace(
            spk_output,
            loss=loss_output.loss,
            metrics={"accuracy": loss_output.accuracy},
            predictions=loss_output.predictions,
        )

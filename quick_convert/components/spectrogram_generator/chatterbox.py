from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...external.chatterbox.s3gen.flow import CausalMaskedDiffWithXvec


class ChatterboxSpectrogramGenerator(nn.Module):
    def __init__(
        self,
        backbone: CausalMaskedDiffWithXvec,
        content_dim: int,
        speaker_dim: int,
        mel_dim: int = 80,
        use_content_encoder: bool = False,
        content_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.backbone = backbone

        self.content_proj = nn.Linear(content_dim, backbone.input_size)

        self.use_content_encoder = use_content_encoder
        self.content_encoder = content_encoder

        self.speaker_proj = nn.Linear(
            speaker_dim,
            backbone.decoder.spk_emb_dim if hasattr(backbone.decoder, "spk_emb_dim") else mel_dim,
        )

    def encode_content(
        self,
        content_features: torch.Tensor,
        content_lengths: torch.Tensor,
    ):
        """
        Converts arbitrary learned speech features into the representation
        expected by the chatterbox backbone encoder.
        """

        x = self.content_proj(content_features)

        if self.use_content_encoder:
            x = self.content_encoder(x, content_lengths)

        return x, content_lengths

    def project_speaker(
        self,
        speaker_embedding: torch.Tensor,
    ):
        speaker_embedding = F.normalize(speaker_embedding, dim=-1)
        return self.speaker_proj(speaker_embedding)

    def compute_loss(
        self,
        content_features: torch.Tensor,
        content_lengths: torch.Tensor,
        target_mel: torch.Tensor,
        target_mel_lengths: torch.Tensor,
        speaker_embedding: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ):
        """
        Thin wrapper around donor compute_loss.
        """

        token_embeddings, token_lengths = self.encode_content(
            content_features,
            content_lengths,
        )

        projected_speaker = self.project_speaker(speaker_embedding)

        batch = {
            # bypass token embedding lookup entirely
            "speech_token": token_embeddings,
            "speech_token_len": token_lengths,
            "speech_feat": target_mel,
            "speech_feat_len": target_mel_lengths,
            "embedding": projected_speaker,
        }

        if cond is not None:
            batch["cond"] = cond

        return self.backbone.compute_loss(
            batch=batch,
            device=target_mel.device,
        )

    @torch.inference_mode()
    def forward(
        self,
        content_features: torch.Tensor,
        content_lengths: torch.Tensor,
        speaker_embedding: torch.Tensor,
        prompt_mel: Optional[torch.Tensor] = None,
        n_timesteps: int = 10,
        cond: Optional[torch.Tensor] = None,
        noised_mels: Optional[torch.Tensor] = None,
        meanflow: bool = False,
    ):
        token_embeddings, token_lengths = self.encode_content(
            content_features,
            content_lengths,
        )

        projected_speaker = self.project_speaker(speaker_embedding)

        B = token_embeddings.size(0)

        if prompt_mel is None:
            prompt_mel = torch.zeros(
                B,
                0,
                self.backbone.output_size,
                device=token_embeddings.device,
                dtype=token_embeddings.dtype,
            )

        prompt_feat_len = torch.zeros(
            B,
            dtype=torch.long,
            device=token_embeddings.device,
        )

        feat, _ = self.backbone.decoder(
            mu=token_embeddings.transpose(1, 2),
            mask=torch.ones(
                B,
                1,
                token_embeddings.size(1),
                device=token_embeddings.device,
                dtype=token_embeddings.dtype,
            ),
            spks=projected_speaker,
            cond=cond,
            n_timesteps=n_timesteps,
            noised_mels=noised_mels,
            meanflow=meanflow,
        )

        return feat

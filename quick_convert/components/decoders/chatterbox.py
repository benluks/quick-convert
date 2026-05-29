from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...external.chatterbox.s3gen.flow import CausalMaskedDiffWithXvec


class ChatterboxSpectrogramGenerator(nn.Module):
    def __init__(
        self,
        flow: CausalMaskedDiffWithXvec,
        content_dim: int,
        speaker_dim: int,
        mel_dim: int = 80,
    ):
        super().__init__()

        self.flow = flow

        self.content_proj = nn.Linear(content_dim, flow.input_size)

        self.speaker_proj = nn.Linear(
            speaker_dim,
            flow.decoder.spk_emb_dim if hasattr(flow.decoder, "spk_emb_dim") else mel_dim,
        )

    def project_speaker(
        self,
        speaker_embedding: torch.Tensor,
    ):
        speaker_embedding = F.normalize(speaker_embedding, dim=-1)
        return self.speaker_proj(speaker_embedding)

    def compute_loss(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        target_mel: torch.Tensor,
        target_mel_lengths: torch.Tensor,
        speaker_embedding: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ):
        """
        Thin wrapper around donor compute_loss.
        """

        projected_speaker = self.project_speaker(speaker_embedding)

        batch = {
            # bypass token embedding lookup entirely
            "speech_token": features,
            "speech_token_len": lengths,
            "speech_feat": target_mel,
            "speech_feat_len": target_mel_lengths,
            "embedding": projected_speaker,
        }

        if cond is not None:
            batch["cond"] = cond

        return self.flow.compute_loss(
            batch=batch,
            device=target_mel.device,
        )

    @torch.inference_mode()
    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        speaker_embedding: torch.Tensor,
        prompt_mel: Optional[torch.Tensor] = None,
        n_timesteps: int = 10,
        cond: Optional[torch.Tensor] = None,
        noised_mels: Optional[torch.Tensor] = None,
        meanflow: bool = False,
    ):

        projected_speaker = self.project_speaker(speaker_embedding)

        B = features.size(0)

        if prompt_mel is None:
            prompt_mel = torch.zeros(
                B,
                0,
                self.flow.output_size,
                device=features.device,
                dtype=features.dtype,
            )

        prompt_feat_len = torch.zeros(
            B,
            dtype=torch.long,
            device=features.device,
        )

        feat, _ = self.flow.decoder(
            mu=features.transpose(1, 2),
            mask=torch.ones(
                B,
                1,
                features.size(1),
                device=features.device,
                dtype=features.dtype,
            ),
            spks=projected_speaker,
            cond=cond,
            n_timesteps=n_timesteps,
            noised_mels=noised_mels,
            meanflow=meanflow,
        )

        return feat

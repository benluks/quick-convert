from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from quick_convert.external.chatterbox.s3gen.utils.mel import mel_spectrogram

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
        self.mel_extractor = mel_spectrogram

    def project_speaker(
        self,
        speaker_embedding: torch.Tensor,
    ):
        speaker_embedding = F.normalize(speaker_embedding, dim=-1)
        return self.speaker_proj(speaker_embedding)

    def _mel_lengths(self, lengths, n_fft=1280, hop_size=320):
        pad = (n_fft - hop_size) // 2
        return ((lengths + 2 * pad - n_fft) // hop_size) + 1

    def compute_loss(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        target_wav: torch.Tensor,
        wav_lens: torch.Tensor,
        sampling_rate: int,
        speaker_embedding: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ):
        """
        Thin wrapper around donor compute_loss.
        """

        n_fft = int(sampling_rate / 12.5)
        hop_size = int(sampling_rate / 50)

        target_mel = self.mel_extractor(
            y=target_wav,
            n_fft=n_fft,
            num_mels=80,
            sampling_rate=sampling_rate,
            hop_size=hop_size,
            win_size=n_fft,
            fmin=0,
            fmax=8000,
            center=False,
        )
        target_mel_lengths = self._mel_lengths(wav_lens, n_fft=n_fft, hop_size=hop_size)

        batch = {
            # bypass token embedding lookup entirely
            "speech_token": features,
            "speech_token_len": lengths,
            "speech_feat": target_mel,
            "speech_feat_len": target_mel_lengths,
            "embedding": speaker_embedding,
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

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from quick_convert.utils.masking import make_padding_mask, masked_loss, trim_to_min

from ...external.chatterbox.bridges.load_vocoder import load_vocoder
from ...external.chatterbox.s3gen.utils.mel import mel_spectrogram
from ...external.chatterbox.s3gen.flow import CausalMaskedDiffWithXvec
from ...external.chatterbox.s3gen.hifigan import HiFTGenerator


class ChatterboxSpectrogramGenerator(nn.Module):
    def __init__(
        self,
        flow: CausalMaskedDiffWithXvec,
        cond_strategy: Literal["rvq", "mel", None] = None,
        device: Optional[torch.device] = None,
        # content_dim: int,
        # speaker_dim: int,
        mel_dim: int = 80,
    ):
        super().__init__()

        self.flow = flow
        self.mel_extractor = mel_spectrogram
        self.cond_strategy = cond_strategy
        self.device = device
        self.vocoder: HiFTGenerator = load_vocoder(device=device)[0]

    def project_speaker(
        self,
        speaker_embedding: torch.Tensor,
    ):
        speaker_embedding = F.normalize(speaker_embedding, dim=-1)
        return self.speaker_proj(speaker_embedding)

    def _mel_lengths(self, lengths, n_fft=1280, hop_size=320):
        pad = (n_fft - hop_size) // 2
        return ((lengths + 2 * pad - n_fft) // hop_size) + 1

    def _compute_mels(self, wav: torch.Tensor, lengths: torch.Tensor, sampling_rate: int, max_len=None):
        n_fft = int(sampling_rate / 12.5)
        hop_size = int(sampling_rate / 50)

        mel = self.mel_extractor(
            y=wav,
            n_fft=n_fft,
            num_mels=80,
            sampling_rate=sampling_rate,
            hop_size=hop_size,
            win_size=n_fft,
            fmin=0,
            fmax=8000,
            center=False,
        )
        mel_lengths = self._mel_lengths(lengths, n_fft=n_fft, hop_size=hop_size)
        if max_len is not None:
            if max_len > mel.shape[-1]:
                mel = F.pad(mel, (0, max_len - mel.shape[-1]))
        return mel, mel_lengths

    def mel2wav(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Use the pretrained Chatterbox vocoder to convert mel spectrograms to waveforms.
        """
        return self.vocoder.inference(mel)[0]

    def compute_loss(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        target_wav: torch.Tensor,
        wav_lens: torch.Tensor,
        sampling_rate: int,
        speaker_embedding: torch.Tensor,
        mask: torch.Tensor = None,
        # cond: Optional[torch.Tensor] = None,
    ):
        """
        Thin wrapper around donor compute_loss.
        """

        target_mel, target_mel_lengths = self._compute_mels(
            target_wav, wav_lens, sampling_rate.item(), max_len=features.shape[1]
        )

        features, target_mel, lengths = trim_to_min(features.transpose(1, 2), target_mel, lengths, target_mel_lengths)

        batch = {
            "speech_token": features.transpose(1, 2),
            "speech_token_len": lengths,
            "speech_feat": target_mel,
            # use legnths output from the trim function above
            "speech_feat_len": lengths,
            "embedding": speaker_embedding,
        }

        mask = make_padding_mask(lengths, max_length=features.shape[-1])

        output = self.flow.compute_loss(
            batch=batch,
            mask=mask,
            device=target_mel.device,
            cond_strategy=self.cond_strategy,
        )

        pred_mel = output["y"]

        mae = masked_loss(F.l1_loss, pred_mel.transpose(1, 2), target_mel.transpose(1, 2), mask=mask)

        return output["loss"], pred_mel, mae

    @torch.inference_mode()
    def forward(
        self,
        feature: torch.Tensor,
        length: torch.Tensor,
        speaker_embedding: torch.Tensor,
        n_timesteps: int = 10,
        # adding max_len because this only suppoorts batch size 1, so in parent class we iterate through batch and
        # call forward on each sample. Instead of unpadding them and then padding them together later, we just
        # pass in the max length for the batch and let the flow handle the masking and padding.
        max_len: Optional[int] = 0,
        cond: Optional[torch.Tensor] = None,
        run_vocoder: bool = False,
    ):
        mel, _ = self.flow.inference(
            token=feature,
            token_len=length,
            embedding=speaker_embedding,
            finalize=True,
            max_feature_len=max_len,
            n_timesteps=n_timesteps,
        )

        return mel

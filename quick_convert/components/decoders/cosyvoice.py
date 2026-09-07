from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from quick_convert.utils.device import DeviceLike, configure_device
from quick_convert.utils.masking import make_padding_mask, trim_to_min

from ...external.cosyvoice.flow.flow import CausalMaskedDiffWithDiT, CausalMaskedDiffWithXvec
from ...external.matcha.utils.audio import mel_spectrogram
from .hift_generator import CosyVoiceHiFTDecoder


@dataclass
class CosyVoiceDecoderOutput:
    flow_state: torch.Tensor | None = None
    loss: torch.Tensor | None = None


class CosyVoiceSpectrogramGenerator(nn.Module):
    VOCODER_SR = 24_000

    def __init__(
        self,
        flow: CausalMaskedDiffWithXvec | CausalMaskedDiffWithDiT,
        # feature dim output by preceding encoder.
        # if encoder outputs discrete tokens, set to `None` and ensure
        # `flow` has `vocab_size` set to an integer value
        feature_dim: int | None,
        cond_strategy: Literal["rvq", "mel"] | None = None,
        device: DeviceLike = None,
        # content_dim: int,
        # speaker_dim: int,
        mel_dim: int = 80,
    ):
        super().__init__()

        self.device = configure_device(device)
        self.flow = flow
        self.mel_extractor = mel_spectrogram
        self.cond_strategy = cond_strategy
        self.vocoder: CosyVoiceHiFTDecoder = CosyVoiceHiFTDecoder.from_pretrained(device=self.device)
        self.input_projection = None
        if flow.vocab_size is None and feature_dim != flow.input_size:
            self.input_projection = nn.Linear(feature_dim, flow.input_size, device=self.device)

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
        if max_len is not None and max_len > mel.shape[-1]:
            mel = F.pad(mel, (0, max_len - mel.shape[-1]))
        return mel, mel_lengths

    def mel2wav(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Use the pretrained CosyVoice vocoder to convert mel spectrograms to waveforms.
        """
        return self.vocoder(mel)

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

        if self.input_projection is not None:
            mask = make_padding_mask(lengths, max_length=features.shape[-1])
            features = self.input_projection(features.transpose(1, 2)) * mask.unsqueeze(-1)
        else:
            features = features.transpose(1, 2)

        batch = {
            "speech_token": features,
            "speech_token_len": lengths,
            "speech_feat": target_mel.transpose(1, 2),
            # use legnths output from the trim function above
            "speech_feat_len": lengths,
            "embedding": speaker_embedding,
        }

        output = self.flow(
            batch=batch,
            # mask=mask,
            device=target_mel.device,
            # cond_strategy=self.cond_strategy,
        )

        return CosyVoiceDecoderOutput(loss=output["loss"])

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
        max_len: int | None = 0,
        cond: torch.Tensor | None = None,
        run_vocoder: bool = False,
    ):

        # because of different feature extractions, sometimes lengths can be
        # 1 frame shorter than the features' actual shape, so we trim to the
        # smaller of the 2 values
        max_length_idx = length.argmax()
        min_max_length = min(feature.shape[1], length[max_length_idx])

        feature = feature[:, :min_max_length]
        length[max_length_idx] = min_max_length

        if self.input_projection is not None:
            mask = make_padding_mask(length, max_length=feature.shape[1])
            feature = self.input_projection(feature) * mask.unsqueeze(-1)

        B, _, D = feature.shape
        mel, _ = self.flow.inference(
            token=feature,
            token_len=length,
            prompt_token=torch.zeros(B, 0, D).long().to(self.device),
            prompt_token_len=torch.zeros((B,)).long().to(self.device),
            prompt_feat=torch.zeros(B, 0, self.flow.output_size).to(self.device),
            prompt_feat_len=torch.zeros((B,)).long().to(self.device),
            embedding=speaker_embedding,
            finalize=True,
            streaming=False,
            # max_feature_len=max_len,
            # n_timesteps=n_timesteps,
        )

        if run_vocoder:
            wav = self.mel2wav(mel)  # mel must be `B, 80, T`
        else:
            wav = None

        return mel, wav

    @torch.inference_mode()
    def inference(self, features, lengths, spk_output):

        mel, _ = self.flow.inference(
            token=features,
            token_len=lengths,
            embedding=spk_output,
            finalize=True,
        )

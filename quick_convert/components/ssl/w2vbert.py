from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModel
from torch.nn.utils.rnn import pad_packed_sequence

from quick_convert.data.types import AudioBatch

from .base import ContentEncoder, ContentFeatures


class W2VBertContentEncoder(ContentEncoder):
    FEATURE_DIM = 1024

    def __init__(
        self,
        model_name: str = "facebook/w2v-bert-2.0",
        sample_rate: int = 16000,
        layer: Optional[int] = None,
        device: str | None = None,
        local_files_only: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(device=device)
        self.model_name = model_name
        self.sample_rate = 16000
        self.layer = layer
        self.local_files_only = local_files_only

        self.processor = AutoFeatureExtractor.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        ).to(self.device)
        self.model.eval()

    def encode_file(self, path: PathLike) -> ContentFeatures:
        path = Path(path)
        wav, sr = torchaudio.load(path)

        # Convert to mono if needed.
        if wav.dim() == 2 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if wav.dim() != 2:
            raise ValueError(f"Expected waveform of shape (channels, time), got {tuple(wav.shape)}")

        # Convert from (1, time) -> (batch=1, time)
        wav = wav.squeeze(0).unsqueeze(0)

        return self.encode_waveforms(wav, sample_rate=sr)

    # def forward(self, batch: AudioBatch):
    #     self.extract_batch(batch)

    def forward(self, batch: AudioBatch):
        if getattr(batch, "waveforms", None) is None:
            raise RuntimeError(
                f"{self.__class__.__name__} only works with loaded audio for now. Please set `load: true` in your dataset config"
            )
        if not (batch.sample_rates == self.sample_rate).all():
            raise RuntimeError(
                f"""Expected sample rates of input audio to be {self.sample_rate}, but got {batch.sample_rates}. 
                Batch resampling within {self.__class__.__name__} is not currently supported. 
                Please set `target_sr={self.sample_rate}` (with `load=True`) in the dataset section of your 
                config to perform resampling at the dataset level."""
            )

        content = self.encode_waveforms(batch.waveforms.to(self.device), batch.lengths.to(self.device))
        return content

    @torch.inference_mode()
    def encode_waveforms(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor | None = None,
        sample_rate: int | None = None,
    ) -> ContentFeatures:
        """
        Args:
            wavs:
                Tensor of shape (batch, time).
            lengths:
                Optional relative lengths in [0, 1], shape (batch,).
                Currently used only to estimate output lengths.
            sample_rate:
                Input waveform sample rate. If None, self.sample_rate is assumed.

        Returns:
            ContentFeatures with values of shape (batch, frames, dim).
        """
        waveforms_list = [waveforms[i, : lengths[i]].cpu() for i in range(waveforms.shape[0])]

        features = self.processor(
            waveforms_list,
            sampling_rate=sample_rate or self.sample_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        ).to(self.device)

        outputs = self.model(
            **features,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states

        if self.layer is None:
            selected = torch.stack(hidden_states[1:], dim=2)  # (batch, frames, layer, dim)
        else:
            selected = hidden_states[self.layer]  # (batch, frames, dim)
        output_lengths = features.attention_mask.sum(1)

        return ContentFeatures(
            values=selected,
            lengths=output_lengths,
            feature_dim=selected.shape[-1],
            representation_type="continuous",
            temporal_granularity="frame",
            backend="transformers",
            model_name=self.model_name,
            layer=self.layer,
        )


class W2VBertContentEncoderDS(ContentEncoder):
    FEATURE_DIM = 1024

    def __init__(
        self,
        model_name: str = "facebook/w2v-bert-2.0",
        sample_rate: int = 16000,
        layer: Optional[int] = None,
        device: str | None = None,
        local_files_only: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(device=device)
        self.model_name = model_name
        self.sample_rate = 16000
        self.layer = layer
        self.local_files_only = local_files_only

        self.processor = AutoFeatureExtractor.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        ).to(self.device)
        self.model.eval()

    def encode_file(self, path: PathLike) -> ContentFeatures:
        path = Path(path)
        wav, sr = torchaudio.load(path)

        # Convert to mono if needed.
        if wav.dim() == 2 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if wav.dim() != 2:
            raise ValueError(f"Expected waveform of shape (channels, time), got {tuple(wav.shape)}")

        # Convert from (1, time) -> (batch=1, time)
        wav = wav.squeeze(0).unsqueeze(0)

        return self.encode_waveforms(wav, sample_rate=sr)

    # def forward(self, batch: AudioBatch):
    #     self.extract_batch(batch)

    def forward(self, batch: AudioBatch):
        if getattr(batch, "waveforms", None) is None:
            raise RuntimeError(
                f"{self.__class__.__name__} only works with loaded audio for now. Please set `load: true` in your dataset config"
            )
        if not (batch.sample_rates == self.sample_rate).all():
            raise RuntimeError(
                f"""Expected sample rates of input audio to be {self.sample_rate}, but got {batch.sample_rates}. 
                Batch resampling within {self.__class__.__name__} is not currently supported. 
                Please set `target_sr={self.sample_rate}` (with `load=True`) in the dataset section of your 
                config to perform resampling at the dataset level."""
            )

        content = self.encode_waveforms(batch.waveforms.to(self.device), batch.lengths.to(self.device))
        return content

    @torch.inference_mode()
    def encode_waveforms(
        self,
        waveforms: torch.Tensor,
        lengths: torch.Tensor | None = None,
        sample_rate: int | None = None,
    ) -> ContentFeatures:
        """
        Args:
            wavs:
                Tensor of shape (batch, time).
            lengths:
                Optional relative lengths in [0, 1], shape (batch,).
                Currently used only to estimate output lengths.
            sample_rate:
                Input waveform sample rate. If None, self.sample_rate is assumed.

        Returns:
            ContentFeatures with values of shape (batch, frames, dim).
        """
        waveforms_list = [waveforms[i, : lengths[i]].cpu() for i in range(waveforms.shape[0])]

        features = self.processor(
            waveforms_list,
            sampling_rate=sample_rate or self.sample_rate,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        ).to(self.device)

        outputs = self.model(
            **features,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states

        if self.layer is None:
            selected = torch.stack(hidden_states[1:], dim=2)  # (batch, frames, layer, dim)
        else:
            selected = hidden_states[self.layer]  # (batch, frames, dim)
        output_lengths = features.attention_mask.sum(1)

        # Pad outputs to the same length
        output, output_lengths = pad_packed_sequence(outputs, batch_first=True, padding_value=0.0)

        # Subsample by a factor of 2 using average pooling
        output = torch.nn.functional.avg_pool1d(output.transpose(1, 2), kernel_size=4, stride=4, padding=0).transpose(
            1, 2
        )
        output_lengths = (output_lengths - 1) // 4 + 1

        return ContentFeatures(
            values=selected,
            lengths=output_lengths,
            feature_dim=selected.shape[-1],
            representation_type="continuous",
            temporal_granularity="frame",
            backend="transformers",
            model_name=self.model_name,
            layer=self.layer,
        )
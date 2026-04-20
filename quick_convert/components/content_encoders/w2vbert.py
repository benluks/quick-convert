from __future__ import annotations

from pathlib import Path

import torch
import torchaudio
from transformers import AutoProcessor, AutoModel

from .base import ContentEncoder, ContentFeatures


class W2VBertContentEncoder(ContentEncoder):
    def __init__(
        self,
        model_name: str = "facebook/w2v-bert-2.0",
        sample_rate: int = 16000,
        layer: int = -1,
        device: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.layer = layer
        self.local_files_only = local_files_only

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        ).to(self.device)
        self.model.eval()

    def encode_file(self, path: str | Path) -> ContentFeatures:
        path = Path(path)
        wav, sr = torchaudio.load(path)

        # Convert to mono if needed.
        if wav.dim() == 2 and wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if wav.dim() != 2:
            raise ValueError(
                f"Expected waveform of shape (channels, time), got {tuple(wav.shape)}"
            )

        # Convert from (1, time) -> (batch=1, time)
        wav = wav.squeeze(0).unsqueeze(0)

        return self.encode_waveforms(wav, sample_rate=sr)

    @torch.inference_mode()
    def encode_waveforms(
        self,
        wavs: torch.Tensor,
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
        if wavs.dim() != 2:
            raise ValueError(
                f"Expected wavs with shape (batch, time), got {tuple(wavs.shape)}"
            )

        sample_rate = sample_rate or self.sample_rate

        if sample_rate != self.sample_rate:
            wavs = torchaudio.functional.resample(
                wavs,
                orig_freq=sample_rate,
                new_freq=self.sample_rate,
            )
            sample_rate = self.sample_rate

        wavs = wavs.detach().cpu()

        # HF audio feature extractors expect a list of 1D arrays/tensors.
        wav_list = [wav for wav in wavs]

        features = self.processor(
            wav_list,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )

        features = {key: value.to(self.device) for key, value in features.items()}

        outputs = self.model(
            **features,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states
        selected = hidden_states[self.layer]  # (batch, frames, dim)

        attention_mask = features.get("attention_mask")

        output_lengths = None
        if attention_mask is not None:
            # Conservative approximation: scale by observed frame reduction.
            input_lengths = attention_mask.sum(dim=1)
            n_input = attention_mask.shape[1]
            n_frames = selected.shape[1]

            output_lengths = torch.floor(
                input_lengths.to(torch.float32) * n_frames / n_input
            ).to(torch.long)

        elif lengths is not None:
            n_frames = selected.shape[1]
            output_lengths = torch.floor(lengths.to(torch.float32) * n_frames).to(
                torch.long
            )

        # frame_hz = None
        # if selected.shape[1] > 0 and wavs.shape[1] > 0:
        #     duration_sec = wavs.shape[1] / sample_rate
        #     if duration_sec > 0:
        #         frame_hz = selected.shape[1] / duration_sec

        return ContentFeatures(
            values=selected,
            lengths=output_lengths,
            # frame_hz=frame_hz,
            feature_dim=selected.shape[-1],
            representation_type="continuous",
            temporal_granularity="frame",
            backend="transformers",
            model_name=self.model_name,
            layer=self.layer,
        )

"""Frozen native-codec quantizer contributions with valid frame lengths."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import replace
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from .base import ContentEncoder, ContentFeatures


class QuantizedContentEncoder(ContentEncoder):
    """Expose native additive RVQ contributions as ``[B, T, Q, D]``.

    Codecs stay frozen and in evaluation mode. Each valid waveform is encoded
    separately so padding cannot change its representations or frame count.
    Outputs are ordinary detached tensors usable by downstream autograd.
    """

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Native waveform sample rate."""

    @property
    @abstractmethod
    def n_codebooks(self) -> int:
        """Number of native quantizer contributions."""

    @abstractmethod
    def _encode_layers(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return ``[T, Q, D]`` for one ``[1, 1, samples]`` waveform."""

    def train(self, mode: bool = True):
        super().train(False)
        return self

    @torch.no_grad()
    def encode_waveforms(self, waveforms, lengths=None, sample_rate=None) -> ContentFeatures:
        if sample_rate is not None and sample_rate != self.sample_rate:
            raise ValueError(f"Expected {self.sample_rate} Hz audio, got {sample_rate}.")
        if waveforms.ndim == 3 and waveforms.shape[1] == 1:
            waveforms = waveforms.squeeze(1)
        if waveforms.ndim != 2 or not waveforms.shape[0]:
            raise ValueError("Expected a nonempty waveform batch shaped [B, samples] or [B, 1, samples].")
        if not waveforms.is_floating_point() or not torch.isfinite(waveforms).all():
            raise ValueError("Waveforms must contain finite floating-point samples.")
        if lengths is None:
            lengths = torch.full((waveforms.shape[0],), waveforms.shape[1], device=waveforms.device)
        if lengths.shape != (waveforms.shape[0],) or lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError("Lengths must be an integer vector with one entry per waveform.")
        if torch.any(lengths <= 0) or torch.any(lengths > waveforms.shape[1]):
            raise ValueError("Waveform lengths must be positive and cannot exceed the padded sample count.")
        self.eval()
        parameter = next(self.parameters(), None)
        device = parameter.device if parameter is not None else self.device
        layers = [
            self._encode_layers(wav[: int(length)].to(device).clone()[None, None]).detach().clone()
            for wav, length in zip(waveforms, lengths, strict=True)
        ]
        if any(value.ndim != 3 or min(value.shape) == 0 or value.shape[1] != self.n_codebooks for value in layers):
            raise RuntimeError("Codec must return nonempty [time, n_codebooks, feature_dim] contributions.")
        values = pad_sequence(layers, batch_first=True)
        if values.ndim != 4 or values.shape[2] != self.n_codebooks:
            raise RuntimeError("Codec must return [time, n_codebooks, feature_dim] contributions.")
        return ContentFeatures(
            values=values,
            lengths=torch.tensor([len(value) for value in layers], device=device),
            feature_dim=values.shape[-1],
            representation_type="quantized",
            temporal_granularity="frame",
            backend=self.BACKEND,
            model_name=self.model_name,
            layer="rvq",
            frame_hz=self.frame_hz,
        )

    def encode_file(self, path: str | Path) -> ContentFeatures:
        import torchaudio

        waveform, rate = torchaudio.load(str(path))
        waveform = waveform.mean(0, keepdim=True)
        if rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, rate, self.sample_rate)
        return self.encode_waveforms(waveform)

    def forward(self, batch) -> ContentFeatures:
        if batch.waveforms is None or batch.lengths is None or batch.sample_rates is None:
            raise ValueError("Native RVQ encoding requires loaded audio, lengths and sample rates.")
        if not torch.all(batch.sample_rates == self.sample_rate):
            raise ValueError(f"Resample audio to {self.sample_rate} Hz before encoding.")
        return self.encode_waveforms(batch.waveforms, batch.lengths)


class DACQuantizedContentEncoder(QuantizedContentEncoder):
    """Frozen pretrained DAC encoder and its original RVQ (decoder discarded)."""

    BACKEND = "dac"

    def __init__(self, model_type: str = "16khz", device: str | None = None):
        super().__init__(device)
        import dac

        model = dac.DAC.load(dac.utils.download(model_type=model_type))
        self.encoder = model.encoder
        self.quantizer = model.quantizer
        self._sample_rate = int(model.sample_rate)
        self.hop_length = int(model.hop_length)
        self._feature_dim = int(model.latent_dim)
        self.model_name = f"dac::{model_type}"
        self.to(self.device).requires_grad_(False).eval()

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def n_codebooks(self):
        return len(self.quantizer.quantizers)

    @property
    def feature_dim(self):
        return self._feature_dim

    @property
    def frame_hz(self):
        return self.sample_rate / self.hop_length

    def _encode_layers(self, waveform):
        from torch.nn.functional import pad

        waveform = pad(waveform, (0, (-waveform.shape[-1]) % self.hop_length))
        residual = self.encoder(waveform)
        layers = []
        for quantizer in self.quantizer.quantizers:
            contribution, *_ = quantizer(residual)
            layers.append(contribution)
            residual = residual - contribution
        return torch.stack(layers, dim=2).squeeze(0).permute(2, 1, 0)


class MimiQuantizedContentEncoder(QuantizedContentEncoder):
    """Frozen Transformers Mimi with native semantic/acoustic projections."""

    BACKEND = "mimi"

    def __init__(self, model_name: str = "kyutai/mimi", device: str | None = None):
        super().__init__(device)
        from transformers import MimiModel

        self.model = MimiModel.from_pretrained(model_name)
        self.model_name = model_name
        self.to(self.device).requires_grad_(False).eval()

    @property
    def sample_rate(self):
        return self.model.config.sampling_rate

    @property
    def n_codebooks(self):
        return self.model.config.num_quantizers

    @property
    def feature_dim(self):
        return self.model.config.hidden_size

    @property
    def frame_hz(self):
        return self.model.config.frame_rate

    def _encode_layers(self, waveform):
        codes = self.model.encode(waveform, num_quantizers=self.n_codebooks, use_streaming=False).audio_codes
        quantizer = self.model.quantizer
        layers = []
        offset = 0
        for branch in (quantizer.semantic_residual_vector_quantizer, quantizer.acoustic_residual_vector_quantizer):
            for layer in branch.layers:
                contribution = layer.decode(codes[:, offset])
                if branch.output_proj is not None:
                    contribution = branch.output_proj(contribution)
                layers.append(contribution)
                offset += 1
        return torch.stack(layers, dim=2).squeeze(0).permute(2, 1, 0)


class SpeechTokenizerQuantizedContentEncoder(QuantizedContentEncoder):
    """Frozen SpeechTokenizer, downloaded by default or loaded from local files.

    ``layer=None`` returns all RVQ contributions as ``[B, T, Q, D]``.
    A zero-based ``layer`` returns that quantizer's embeddings as ``[B, T, D]``;
    ``layer=0`` is the first-level semantic content used by DiffAnon.
    """

    BACKEND = "speechtokenizer"
    MODEL_REPO = "fnlp/SpeechTokenizer"
    MODEL_DIRECTORY = "speechtokenizer_hubert_avg"

    def __init__(
        self,
        config_path: str | None = None,
        checkpoint_path: str | None = None,
        device: str | None = None,
        layer: int | None = None,
    ):
        super().__init__(device)
        if layer is not None and (isinstance(layer, bool) or not isinstance(layer, int) or layer < 0):
            raise ValueError("layer must be a nonnegative integer RVQ index or None.")
        if (config_path is None) != (checkpoint_path is None):
            raise ValueError(
                "Provide both config_path and checkpoint_path, or omit both to download the default model."
            )
        from speechtokenizer import SpeechTokenizer

        if config_path is None:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(self.MODEL_REPO, f"{self.MODEL_DIRECTORY}/config.json")
            checkpoint_path = hf_hub_download(self.MODEL_REPO, f"{self.MODEL_DIRECTORY}/SpeechTokenizer.pt")
        self.model = SpeechTokenizer.load_from_checkpoint(config_path, checkpoint_path)
        if layer is not None and layer >= self.n_codebooks:
            raise ValueError(f"layer must be less than the checkpoint's {self.n_codebooks} codebooks.")
        self.layer = layer
        self.model_name = str(checkpoint_path)
        self.to(self.device).requires_grad_(False).eval()

    @property
    def sample_rate(self):
        return self.model.sample_rate

    @property
    def n_codebooks(self):
        return self.model.n_q

    @property
    def feature_dim(self):
        return self.model.quantizer.dimension

    @property
    def frame_hz(self):
        return self.sample_rate / self.model.downsample_rate

    def _encode_layers(self, waveform):
        layers = self.model.forward_feature(waveform, layers=list(range(self.n_codebooks)))
        return torch.stack(layers, dim=2).squeeze(0).permute(2, 1, 0)

    def encode_waveforms(self, waveforms, lengths=None, sample_rate=None) -> ContentFeatures:
        features = super().encode_waveforms(waveforms, lengths, sample_rate)
        if self.layer is None:
            return features
        return replace(features, values=features.values[:, :, self.layer], layer=self.layer)

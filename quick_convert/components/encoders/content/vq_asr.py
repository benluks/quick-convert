"""Frozen quantized content from a trained VQ-ASR system."""

from pathlib import Path

import torch
from torch import nn

from quick_convert.components.encoders.content.base import ContentEncoder, ContentFeatures
from quick_convert.data import AudioBatch
from quick_convert.utils.masking import make_padding_mask


class VQASRContentEncoder(ContentEncoder):
    """Reuse a restored VQ-ASR bottleneck without running its CTC head.

    The source encoder, layer fusion and quantizer are always frozen and in
    evaluation mode, including when a parent enters training mode. Outputs are
    projected quantized vectors, never continuous pre-quantization latents.

    Takes ownership of the supplied system's content encoder, layer fusion and
    quantizer: these shared modules are frozen in place. Load a separate source
    system if it must remain trainable elsewhere. The CTC head is not retained.
    """

    def __init__(self, system) -> None:
        from quick_convert.systems.asr import VQASRSystem

        if not isinstance(system, VQASRSystem):
            raise TypeError("VQASRContentEncoder requires a restored VQASRSystem.")
        if "content" not in system.online_encoders:
            raise ValueError("The VQ-ASR system needs an online content encoder.")
        content = system.online_encoders["content"]
        super().__init__(device=content.device)
        self.content_encoder = content
        self.layer_fusion = system.layer_fusion
        self.quantizer = system.quantizer
        self.requires_grad_(False)
        self.eval()

    @property
    def feature_dim(self) -> int:
        return self.quantizer.input_dim

    @property
    def sample_rate(self) -> int:
        return self.content_encoder.sample_rate

    @property
    def frame_hz(self) -> float:
        return self.content_encoder.frame_hz

    def train(self, mode: bool = True):
        # EMA buffers update in training mode even with gradients disabled.
        return super().train(False)

    def _quantize(self, content: ContentFeatures) -> ContentFeatures:
        features = self.layer_fusion(content.values)
        lengths = (
            content.lengths
            if isinstance(self.layer_fusion, nn.Identity)
            else self.layer_fusion.output_lengths(content.lengths)
        )
        mask = make_padding_mask(lengths, max_length=features.shape[1])
        output = self.quantizer(features.transpose(1, 2), mask)
        return ContentFeatures(
            values=output.z_q.transpose(1, 2),
            lengths=self.quantizer.output_lengths(lengths),
            feature_dim=self.feature_dim,
            representation_type="quantized",
            temporal_granularity="frame",
            backend="quick_convert",
            model_name="vq_asr",
            layer=None,
            frame_hz=self.frame_hz,
        )

    @torch.no_grad()
    def forward(self, batch: AudioBatch) -> ContentFeatures:
        return self._quantize(self.content_encoder(batch))

    @torch.no_grad()
    def encode_file(self, path: str | Path) -> ContentFeatures:
        return self._quantize(self.content_encoder.encode_file(path))

    @torch.no_grad()
    def encode_waveforms(self, waveforms, lengths=None, sample_rate=None) -> ContentFeatures:
        if sample_rate is not None and sample_rate != self.sample_rate:
            raise ValueError(f"Expected {self.sample_rate} Hz waveforms.")
        return self._quantize(self.content_encoder.encode_waveforms(waveforms, lengths))

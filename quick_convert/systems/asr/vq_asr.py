from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from quick_convert.components.encoders import LinguisticCTCHead
from quick_convert.components.layers import LayerWeightedSum
from quick_convert.components.layers.rvq import BaseResidualVectorQuantizer, RVQOutput
from quick_convert.components.mixins.resource import OnlineResourceMixin
from quick_convert.components.ssl import ContentEncoder
from quick_convert.data import AudioBatch
from quick_convert.systems._content import module_output_lengths, quantize_content, transform_content
from quick_convert.systems.base import InferenceSystem
from quick_convert.utils.masking import make_padding_mask
from quick_convert.utils.sequence_validation import validate_feature_sequence


@dataclass
class VQASRResult:
    """Inference output from :class:`VQASRSystem`.

    Shapes:
        logits: ``[B, T, V]`` CTC logits.
        lengths: ``[B]`` valid logit lengths.
        contextual: ``[B, T, D]`` representations consumed by the CTC head.
        z_q: ``[B, D, T]`` summed codebook embeddings.
        latents: ``[B, Q * codebook_dim, T]`` pre-quantization residual inputs.
        codes: ``[B, Q, T]`` discrete codebook indices.
    """

    logits: torch.Tensor
    lengths: torch.Tensor
    contextual: torch.Tensor
    quantizer: RVQOutput

    @property
    def z_q(self) -> torch.Tensor:
        return self.quantizer.z_q

    @property
    def latents(self) -> torch.Tensor:
        return self.quantizer.latents

    @property
    def codes(self) -> torch.Tensor:
        return self.quantizer.codes


class VQASRSystem(OnlineResourceMixin, InferenceSystem):
    """Inference-ready quantized ASR system independent of Lightning."""

    def __init__(
        self,
        quantizer: BaseResidualVectorQuantizer,
        ctc_head: LinguisticCTCHead,
        layer_fusion: LayerWeightedSum | None = None,
        post_quantization_network: nn.Module | None = None,
        online_encoders: dict[str, ContentEncoder] | None = None,
        trainable_online_encoders: tuple[str, ...] = (),
        use_latents: bool = True,
    ) -> None:
        super().__init__()

        self.quantizer = quantizer
        self.ctc_head = ctc_head
        self.layer_fusion = layer_fusion or nn.Identity()
        self.post_quantization_network = post_quantization_network
        self.configure_online_encoders(online_encoders, trainable_online_encoders)
        self.use_latents = use_latents

        n_codebooks = getattr(quantizer, "n_codebooks", None)
        if n_codebooks is not None and n_codebooks != 1:
            raise ValueError(
                f"VQASRSystem requires a single active codebook; configured quantizer has n_codebooks={n_codebooks}."
            )

    def encode_features(
        self, values: torch.Tensor, lengths: torch.Tensor, *, metadata: Mapping[str, Any] | None = None
    ) -> VQASRResult:
        """Run VQ-ASR from content tensors and optional source metadata."""
        features, feature_lengths = transform_content(values, lengths, self.layer_fusion, metadata=metadata, trim=True)
        quantizer_output, quantized_lengths = quantize_content(features, feature_lengths, self.quantizer)
        representation = quantizer_output.latents if self.use_latents else quantizer_output.z_q
        representation = representation.transpose(1, 2)

        contextual = representation
        output_lengths = quantized_lengths
        if self.post_quantization_network is not None:
            padding_mask = make_padding_mask(quantized_lengths, max_length=representation.shape[1])
            contextual = self.post_quantization_network(representation, padding_mask)
            output_lengths = module_output_lengths(
                self.post_quantization_network,
                quantized_lengths,
                "post_quantization_network",
            )

        validate_feature_sequence(contextual, output_lengths, name="Post-quantization content")
        output_mask = make_padding_mask(output_lengths, max_length=contextual.shape[1])
        head_output = self.ctc_head.predict(
            contextual,
            lengths=output_lengths,
            padding_mask=output_mask,
        )
        logits = head_output.states.get("logits", head_output.predictions)
        if not isinstance(logits, torch.Tensor):
            raise TypeError("The VQ-ASR CTC head must return tensor logits.")

        return VQASRResult(
            logits=logits,
            lengths=output_lengths,
            contextual=contextual,
            quantizer=quantizer_output,
        )

    def forward(self, batch: AudioBatch) -> VQASRResult:
        content = self.get_resource(batch, "content")
        if content.lengths is None:
            raise ValueError("VQ-ASR content features require valid lengths.")
        return self.encode_features(content.values, content.lengths, metadata=content.metadata)

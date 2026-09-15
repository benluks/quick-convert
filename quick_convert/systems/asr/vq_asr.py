from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from quick_convert.components.encoders import LinguisticCTCHead
from quick_convert.components.layers import LayerWeightedSum
from quick_convert.components.layers.rvq import BaseResidualVectorQuantizer, RVQOutput
from quick_convert.components.mixins.resource import OnlineResourceMixin
from quick_convert.components.ssl import ContentEncoder
from quick_convert.data import AudioBatch
from quick_convert.utils.masking import make_padding_mask


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


class VQASRSystem(OnlineResourceMixin, nn.Module):
    """Inference-ready quantized ASR system independent of Lightning."""

    def __init__(
        self,
        quantizer: BaseResidualVectorQuantizer,
        ctc_head: LinguisticCTCHead,
        layer_fusion: LayerWeightedSum | None = None,
        post_quantization_network: nn.Module | None = None,
        online_encoders: dict[str, ContentEncoder] | None = None,
        use_latents: bool = True,
    ) -> None:
        super().__init__()

        self.quantizer = quantizer
        self.ctc_head = ctc_head
        self.layer_fusion = layer_fusion or nn.Identity()
        self.post_quantization_network = post_quantization_network
        self.online_encoders = nn.ModuleDict(online_encoders or {})
        self.use_latents = use_latents

        self.online_encoders.requires_grad_(False)
        self.online_encoders.eval()

        n_codebooks = getattr(quantizer, "n_codebooks", None)
        if n_codebooks is not None and n_codebooks != 1:
            raise ValueError(
                f"VQASRSystem requires a single active codebook; configured quantizer has n_codebooks={n_codebooks}."
            )

    @staticmethod
    def _module_output_lengths(module: nn.Module, lengths: torch.Tensor, role: str) -> torch.Tensor:
        if isinstance(module, nn.Identity):
            return lengths
        output_lengths = getattr(module, "output_lengths", None)
        if not callable(output_lengths):
            raise TypeError(f"VQ-ASR {role} modules must expose output_lengths(input_lengths).")
        return output_lengths(lengths)

    def encode_features(self, values: torch.Tensor, lengths: torch.Tensor) -> VQASRResult:
        """Run VQ-ASR from batched content features and their valid lengths."""
        features = self.layer_fusion(values)
        feature_lengths = self._module_output_lengths(self.layer_fusion, lengths, "layer_fusion")

        if feature_lengths.shape != (features.shape[0],):
            raise ValueError(
                f"Expected one content length per batch item, got {tuple(feature_lengths.shape)} "
                f"for batch size {features.shape[0]}."
            )
        if torch.any(feature_lengths > features.shape[1]):
            raise ValueError("A content length exceeds the padded feature time dimension.")

        max_length = int(feature_lengths.max().item())
        features = features[:, :max_length]
        padding_mask = make_padding_mask(feature_lengths, max_length=max_length)

        quantizer_output: RVQOutput = self.quantizer(features.transpose(1, 2), padding_mask)
        representation = quantizer_output.latents if self.use_latents else quantizer_output.z_q
        representation = representation.transpose(1, 2)
        quantized_lengths = self.quantizer.output_lengths(feature_lengths)

        contextual = representation
        output_lengths = quantized_lengths
        if self.post_quantization_network is not None:
            contextual = self.post_quantization_network(representation, padding_mask)
            output_lengths = self._module_output_lengths(
                self.post_quantization_network,
                quantized_lengths,
                "post_quantization_network",
            )

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
        return self.encode_features(content.values, content.lengths)

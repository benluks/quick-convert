# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#               2022 Xingchen Song (sxc19@mails.tsinghua.edu.cn)
#               2024 Alibaba Inc (Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Encoder definition."""

import torch

from quick_convert.external.cosyvoice.transformer.convolution import ConvolutionModule
from quick_convert.external.cosyvoice.transformer.encoder_layer import ConformerEncoderLayer
from quick_convert.external.cosyvoice.transformer.positionwise_feed_forward import PositionwiseFeedForward
from quick_convert.external.cosyvoice.transformer.upsample_encoder import PreLookaheadLayer, Upsample1D
from quick_convert.external.cosyvoice.utils.mask import add_optional_chunk_mask, make_pad_mask
from quick_convert.external.cosyvoice.utils.transformer_classes import (
    COSYVOICE_ACTIVATION_CLASSES,
    COSYVOICE_ATTENTION_CLASSES,
    COSYVOICE_EMB_CLASSES,
    COSYVOICE_SUBSAMPLE_CLASSES,
)


class UpsampleConformerEncoder(torch.nn.Module):
    """Configurable CosyVoice-style stacks with optional temporal upsampling.

    ``num_blocks`` counts only the first stack; ``post_upsample_blocks`` explicitly
    adds a second stack. Defaults use one six-block stack without resampling,
    lookahead, or a second projection. Factor one bypasses the resampler entirely.
    A lookahead of zero still enables a learned causal residual convolution;
    None removes it. ``causal`` controls only the blocks' convolution branches,
    not attention; ``streaming=True`` selects fixed chunk attention.

    Adapted from FunAudioLLM/CosyVoice's transformer/upsample_encoder.py,
    itself attributed to ESPnet. Original Apache-2.0 notices are retained.
    Underlying attention and Conformer blocks remain vendored implementation
    details. Original parameter names are retained for strict checkpoint loading
    when the same architecture is selected.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "linear",
        pos_enc_layer_type: str = "rel_pos_espnet",
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        global_cmvn: torch.nn.Module = None,
        macaron_style: bool = True,
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        key_bias: bool = True,
        post_upsample_blocks: int = 0,
        upsample_factor: int = 1,
        lookahead_frames: int | None = None,
        second_projection: bool = False,
    ):
        super().__init__()
        if input_layer not in {"linear", "linear_legacy"}:
            raise ValueError("input_layer must be linear or linear_legacy; subsampling is unsupported")
        if num_blocks < 0 or post_upsample_blocks < 0:
            raise ValueError("Layer counts must be nonnegative")
        if type(upsample_factor) is not int or upsample_factor < 1:
            raise ValueError("upsample_factor must be a positive integer")
        if lookahead_frames is not None and lookahead_frames < 0:
            raise ValueError("lookahead_frames must be nonnegative or None")
        if output_size <= 0 or attention_heads <= 0 or output_size % attention_heads:
            raise ValueError("output_size must be positive and divisible by attention_heads")
        self._output_size = output_size

        self.global_cmvn = global_cmvn
        self.embed = COSYVOICE_SUBSAMPLE_CLASSES[input_layer](
            input_size,
            output_size,
            dropout_rate,
            COSYVOICE_EMB_CLASSES[pos_enc_layer_type](output_size, positional_dropout_rate),
        )

        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-5)
        self.static_chunk_size = static_chunk_size
        self.upsample_factor = upsample_factor
        activation = COSYVOICE_ACTIVATION_CLASSES[activation_type]()
        # self-attention module definition
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
            key_bias,
        )
        # feed-forward module definition
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )
        # convolution module definition
        convolution_layer_args = (output_size, cnn_module_kernel, activation, cnn_module_norm, causal)
        self.pre_lookahead_layer = (
            PreLookaheadLayer(output_size, output_size, lookahead_frames) if lookahead_frames is not None else None
        )
        self.encoders = torch.nn.ModuleList(
            [
                ConformerEncoderLayer(
                    output_size,
                    COSYVOICE_ATTENTION_CLASSES[selfattention_layer_type](*encoder_selfattn_layer_args),
                    PositionwiseFeedForward(*positionwise_layer_args),
                    PositionwiseFeedForward(*positionwise_layer_args) if macaron_style else None,
                    ConvolutionModule(*convolution_layer_args) if use_cnn_module else None,
                    dropout_rate,
                    normalize_before,
                )
                for _ in range(num_blocks)
            ]
        )
        self.up_layer = Upsample1D(output_size, output_size, upsample_factor) if upsample_factor > 1 else None
        self.up_embed = (
            COSYVOICE_SUBSAMPLE_CLASSES[input_layer](
                output_size,
                output_size,
                dropout_rate,
                COSYVOICE_EMB_CLASSES[pos_enc_layer_type](output_size, positional_dropout_rate),
            )
            if second_projection
            else None
        )

        self.up_encoders = torch.nn.ModuleList(
            [
                ConformerEncoderLayer(
                    output_size,
                    COSYVOICE_ATTENTION_CLASSES[selfattention_layer_type](*encoder_selfattn_layer_args),
                    PositionwiseFeedForward(*positionwise_layer_args),
                    PositionwiseFeedForward(*positionwise_layer_args) if macaron_style else None,
                    ConvolutionModule(*convolution_layer_args) if use_cnn_module else None,
                    dropout_rate,
                    normalize_before,
                )
                for _ in range(post_upsample_blocks)
            ]
        )

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        """Return encoder output lengths for padded batched inference."""
        return input_lengths * self.upsample_factor

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        context: torch.Tensor | None = None,
        streaming: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return features (B, T * factor, D) and a valid-frame mask (B, 1, T * factor).

        Streaming uses fixed chunks with unlimited left context. Context contains
        exactly the configured lookahead frames and is accepted only in eval mode.
        """
        if context is None:
            context = xs.new_empty((0, 0, 0))
        if context.numel() and self.pre_lookahead_layer is None:
            raise ValueError("context requires lookahead to be enabled")
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        if context.size(1) != 0:
            assert self.training is False, "you have passed context, make sure that you are running inference mode"
            context_masks = torch.ones(1, 1, context.size(1)).to(masks)
            context, _, _ = self.embed(context, context_masks, offset=xs.size(1))
        mask_pad = masks  # (B, 1, T/subsample_rate)
        chunk_masks = add_optional_chunk_mask(
            xs, masks, False, False, 0, self.static_chunk_size if streaming is True else 0, -1
        )
        # lookahead + conformer encoder
        if self.pre_lookahead_layer is not None:
            xs = self.pre_lookahead_layer(xs, context=context)
        xs = self.forward_layers(xs, chunk_masks, pos_emb, mask_pad)

        # upsample + conformer encoder
        xs = xs.transpose(1, 2).contiguous()
        if self.up_layer is not None:
            xs, xs_lens = self.up_layer(xs, xs_lens)
        xs = xs.transpose(1, 2).contiguous()
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        if self.up_embed is not None:
            xs, pos_emb, masks = self.up_embed(xs, masks)
        else:
            _, pos_emb = self.embed.pos_enc(xs)
        mask_pad = masks  # (B, 1, T/subsample_rate)
        chunk_masks = add_optional_chunk_mask(
            xs, masks, False, False, 0, self.static_chunk_size * self.upsample_factor if streaming is True else 0, -1
        )
        xs = self.forward_up_layers(xs, chunk_masks, pos_emb, mask_pad)

        if self.normalize_before:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks

    def forward_layers(
        self, xs: torch.Tensor, chunk_masks: torch.Tensor, pos_emb: torch.Tensor, mask_pad: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        return xs

    def forward_up_layers(
        self, xs: torch.Tensor, chunk_masks: torch.Tensor, pos_emb: torch.Tensor, mask_pad: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.up_encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        return xs

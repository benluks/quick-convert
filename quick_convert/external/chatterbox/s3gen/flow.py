# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import random
from typing import Dict, Literal, Optional

logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
from torch.nn import functional as F
from .utils.mask import make_pad_mask
from .configs import CFM_PARAMS
from omegaconf import DictConfig
from .flow_matching import CausalConditionalCFM


logger = logging.getLogger(__name__)


def _repeat_batch_dim(tnsr, B, ndim):
    "repeat batch dimension if it's equal to 1"
    if tnsr is not None:
        # add missing batch dim if needed
        while tnsr.ndim < ndim:
            tnsr = tnsr[None]
        # repeat batch dim as needed
        if B > 1 and tnsr.size(0) == 1:
            tnsr = tnsr.repeat(B, *([1] * (ndim - 1)))
        assert tnsr.ndim == ndim, f"Expected {ndim=}, got {tnsr.ndim=}"
    return tnsr


class CausalMaskedDiffWithXvec(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 512,
        output_size: int = 80,
        spk_embed_dim: int = 192,
        output_type: str = "mel",
        vocab_size: int = 6561,
        input_frame_rate: int = 25,
        only_mask_loss: bool = True,
        token_mel_ratio: int = 2,
        pre_lookahead_len: int = 3,
        encoder: torch.nn.Module = None,
        decoder: CausalConditionalCFM = None,
        decoder_conf: Dict = {
            "in_channels": 240,
            "out_channel": 80,
            "spk_emb_dim": 80,
            "n_spks": 1,
            "cfm_params": DictConfig(
                {
                    "sigma_min": 1e-06,
                    "solver": "euler",
                    "t_scheduler": "cosine",
                    "training_cfg_rate": 0.2,
                    "inference_cfg_rate": 0.7,
                    "reg_loss_type": "l1",
                }
            ),
            "decoder_params": {
                "channels": [256, 256],
                "dropout": 0.0,
                "attention_head_dim": 64,
                "n_blocks": 4,
                "num_mid_blocks": 12,
                "num_heads": 8,
                "act_fn": "gelu",
            },
        },
        mel_feat_conf: Dict = {
            "n_fft": 1024,
            "num_mels": 80,
            "sampling_rate": 22050,
            "hop_size": 256,
            "win_size": 1024,
            "fmin": 0,
            "fmax": 8000,
        },
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.mel_feat_conf = mel_feat_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        logging.info(f"input frame rate={self.input_frame_rate}")

        if vocab_size is not None:
            self.input_embedding = nn.Embedding(vocab_size, input_size)
        else:
            self.input_embedding = None
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.encoder = encoder
        self.encoder_proj = torch.nn.Linear(self.encoder.output_size(), output_size)
        self.decoder = decoder
        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio
        self.pre_lookahead_len = pre_lookahead_len

    # NOTE: copied in from cosyvoice repo
    def compute_loss(
        self,
        batch: dict,
        mask: torch.Tensor,
        device: torch.device,
        cond_strategy: Literal["rvq", "mel", None] = "rvq",
    ) -> Dict[str, Optional[torch.Tensor]]:
        token = batch["speech_token"].to(device)
        token_len = batch["speech_token_len"].to(device)
        feat = batch["speech_feat"].to(device)  # (B, 80, T)
        feat_len = batch["speech_feat_len"].to(device)
        embedding = batch["embedding"].to(device)

        # NOTE unified training, static_chunk_size > 0 or = 0
        # streaming = True if random.random() < 0.5 else False

        # xvec projection
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        # mask = (~make_pad_mask(token_len)).to(device)

        if self.input_embedding is not None:
            # if `token` is token indices, project them by input embedding
            token = self.input_embedding(torch.clamp(token, min=0)) * mask.float().unsqueeze(-1)  # (B, T, emb)

        # just for cleanliness, bind `h_lengths` to a value. Originally it would
        # have been returned by the `self.encoder` forward pass.
        h_lengths = None
        h = self.encoder(token, mask)  # (B, T, C)
        h = self.encoder_proj(h)

        # get conditions
        # TODO: replace conds with our own conditioning. In the original implementation,
        # they say "for half the samples, give up to 30% hint with the orginal mel spectrogram"
        # For our examples, we want something more like: prefix with the quantized strams from the
        # crvq encoder.

        if cond_strategy == "mel":
            conds = torch.zeros(feat.shape, device=token.device)
            for i, j in enumerate(feat_len):
                if random.random() < 0.5:
                    continue
                index = random.randint(0, int(0.3 * j))
                conds[i, :, :index] = feat[i, :, :index]
        elif cond_strategy == "rvq":
            conds = self.rvq_cond_proj(token).transpose(1, 2)  # (B, T, C) -> (B, T, mel_dim) -> (B, mel_dim, T)
        elif cond_strategy is None:
            conds = None

        # got rid of this part where h is masked because h is a different lengths
        # from feat (since the original tts upsamples)
        if h.shape[1] != token.shape[1]:
            try:
                mask = (~make_pad_mask(h_lengths.sum(dim=-1).squeeze(dim=1))).to(h)
            except RuntimeError:
                logger.error("It looks like your encoder doesn't output lengths in the forward pass.")

        loss, y = self.decoder.compute_loss(
            feat.contiguous()[..., : token.shape[1]],  # (B, mel_dim, T) -> (B, mel_dim, T')
            mask.unsqueeze(1),
            h.transpose(1, 2).contiguous(),
            embedding,
            cond=conds,
            # streaming=streaming,
        )
        return {"loss": loss, "y": y}

    @torch.inference_mode()
    def inference(
        self,
        token,
        token_len,
        embedding,
        finalize,
        max_feature_len=0,
        prompt_feat=None,
        prompt_feat_len=None,
        prompt_token=None,
        prompt_token_len=None,
        n_timesteps=10,
        noised_mels=None,
        meanflow=False,
        cond_strategy: Literal["rvq", "mel", None] = None,
    ):
        # token: (B, n_toks)
        # token_len: (B,)
        B = token.size(0)
        assert B == 1, "Only batch size of 1 is supported for inference, but got {}".format(B)

        # xvec projection
        embedding = torch.atleast_2d(embedding)
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)  # (1 or B, emb_dim)

        if prompt_feat is not None and prompt_token is not None:
            # adjust shapes (batching logic)
            prompt_token = _repeat_batch_dim(prompt_token, B, ndim=2)  # (B, n_prompt)
            prompt_token_len = _repeat_batch_dim(prompt_token_len, B, ndim=1)  # (B,)
            prompt_feat = _repeat_batch_dim(prompt_feat, B, ndim=3)  # (B, n_feat, feat_dim=80)
            prompt_feat_len = _repeat_batch_dim(prompt_feat_len, B, ndim=1)  # (B,) or None

            token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        embedding = _repeat_batch_dim(embedding, B, ndim=2)  # (B, emb_dim)

        # concat text and prompt_text

        mask = make_pad_mask(token_len, max_len=max_feature_len).to(token.device)  # (B, T)
        if self.input_embedding is not None:
            if (token >= self.vocab_size).any():
                logger.error(
                    f"{token.max()}>{self.vocab_size}\n out-of-range special tokens found in flow, fix inputs!"
                )
            token = self.input_embedding(token.long()) * (~mask).unsqueeze(-1)

        # maybe the conversion should start at this point

        # text encode
        h_masks = None
        h = self.encoder(token, mask)
        if finalize is False:
            h = h[:, : -self.pre_lookahead_len * self.token_mel_ratio]

        mel_len1 = 0

        if self.input_embedding is not None:
            h_lengths = h_masks.sum(dim=-1).squeeze(dim=-1)
            mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]

        h = self.encoder_proj(h)

        # # get conditions
        if cond_strategy == "mel":
            conds = torch.zeros([B, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)
            conds[:, :mel_len1] = prompt_feat
            conds = conds.transpose(1, 2)
            mask = (~make_pad_mask(h_lengths)).unsqueeze(1).to(h)

            if mask.shape[0] != B:
                mask = mask.repeat(B, 1, 1)
        elif cond_strategy is None:
            conds = None
            mask = (~mask).unsqueeze(1)

        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask,
            spks=embedding,
            cond=conds,
            n_timesteps=n_timesteps,
            noised_mels=noised_mels,
            meanflow=meanflow,
        )
        feat = feat[:, :, mel_len1:]
        # assert feat.shape[2] == mel_len2
        return feat, None  # NOTE jrm: why are they returning None here?

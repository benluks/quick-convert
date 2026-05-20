import math
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn

from quick_convert.components.ssl.w2vbert import W2VBertContentEncoder
from quick_convert.components.layers import ParallelConformerEncoder, ResidualVectorQuantizer

class RVQDisentangler(nn.Module):
    """
    Disentanglement model with three components:

      1. W2VBertContentEncoder — waveform (B, T) -> (B, T, L, F)
      2. ParallelConformerEncoder (content encoder) — (B, T, L, F) -> (B, T, F)
      3. ResidualVectorQuantizer (RVQ) — (B·T, F) -> z_q, indices, perplexity, remainders

    Two entries of the RVQ remainder list are selected at the indices given by
    ``rvq_spk_idx`` and ``rvq_pros_idx``.  ``remainder_list[0]`` is the
    original content vector; ``remainder_list[k]`` (k ≥ 1) is the residual
    after the k-th codebook.

    Args:
        feature_extractor:  Instantiated W2VBertContentEncoder.
        content_encoder:    Instantiated ParallelConformerEncoder.
        rvq:                Instantiated ResidualVectorQuantizer.
        rvq_spk_idx:        Index into remainder_list for speaker features.
        rvq_pros_idx:       Index into remainder_list for prosody features.
    """

    def __init__(
        self,
        feature_extractor: W2VBertContentEncoder, # TODO - ideally this would be an interface that could support multiple SSL models
        content_encoder: ParallelConformerEncoder,
        rvq: ResidualVectorQuantizer,
        rvq_spk_idx: int = 1,
        rvq_pros_idx: int = 2,
        prosody_output_dim: int = 0,
        emotion_output_dim: int = 0,
        speaker_output_dim: int = 0,
        linguistic_output_dim: int = 0,
    ):
        super().__init__()

        self.rvq_spk_idx = rvq_spk_idx
        self.rvq_pros_idx = rvq_pros_idx
        self.feature_extractor = feature_extractor
        self.content_encoder = content_encoder
        self.rvq = rvq

        # For KL losses
        self.prosody_head = nn.Linear(content_encoder.output_dim, prosody_output_dim)
        self.emotion_head = nn.Linear(content_encoder.output_dim, emotion_output_dim)

        # We could use Massa's model here
        self.speaker_head = nn.Linear(content_encoder.output_dim, speaker_output_dim)

        # For CTC loss on the linguistic content
        self.linguistic_head = nn.Linear(content_encoder.output_dim, linguistic_output_dim)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_padding_mask(
        self,
        frame_lengths: torch.Tensor,  # (B,)
        max_len: int,
    ) -> torch.Tensor:
        """Returns (B, T) bool mask — True marks padding positions."""
        idx = torch.arange(max_len, device=frame_lengths.device)   # (T,)
        return idx.unsqueeze(0) >= frame_lengths.unsqueeze(1)       # (B, T)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        waveform: torch.Tensor,   # (B, T_samples)
        lengths: torch.Tensor,    # (B,)  — number of valid samples per item
    ) -> Tuple[
        torch.Tensor,             # z_q            (B, T, F)
        List[torch.Tensor],       # indices_list   [num_codebooks × (B·T,)]
        List[float],              # perplexity_list
        List[torch.Tensor],       # remainder_list (flat, length = num_codebooks + 1)
        torch.Tensor,             # spk_remainder  (B, T, F)
        torch.Tensor,             # pros_remainder (B, T, F)
    ]:
        """
        Args:
            waveform: Raw waveform tensor of shape (B, T_samples).
            lengths:  Number of valid waveform samples per batch item, shape (B,).

        Returns:
            z_q:             Differentiable RVQ reconstruction, shape (B, T, F).
            indices_list:    Per-codebook discrete indices.
            perplexity_list: Per-codebook perplexity values.
            remainder_list:  Raw (flattened B·T) remainder tensors from the RVQ.
                             Index 0 is the original content vector; index k is the
                             residual after the k-th codebook.
            spk_remainder:   remainder_list[rvq_spk_idx] reshaped to (B, T, F).
            pros_remainder:  remainder_list[rvq_pros_idx] reshaped to (B, T, F).
        """
        B, T_samples = waveform.shape

        # 1. Build sample-level padding mask for the SSL model
        sample_idx = torch.arange(T_samples, device=waveform.device)
        sample_mask = (sample_idx.unsqueeze(0) < lengths.unsqueeze(1)).long()  # (B, T_samples)

        # 2. Feature extraction: (B, T_samples) -> (B, T_frames, L, F)
        features = self.feature_extractor(waveform, attention_mask=sample_mask)
        T_frames = features.shape[1]

        # 3. Frame-level padding mask for the content encoder
        frame_lengths = self.feature_extractor.get_output_lengths(lengths)
        frame_lengths = frame_lengths.clamp(max=T_frames)
        padding_mask = self._make_padding_mask(frame_lengths, T_frames)  # (B, T_frames)

        # 4. Content encoder: (B, T, L, F) -> (B, T, F)
        content = self.content_encoder(features, padding_mask=padding_mask)

        # 5. Flatten temporal dim for the RVQ which expects (N, D)
        B, T, F = content.shape
        content = content.transpose(1,2) # (B, F, T)

        # 6. Residual VQ
        #z_q, z_qs, codes, latents, commitment_loss, codebook_loss
        z_q_flat, z_quantized, _, _, commitment_loss, codebook_loss = self.rvq(content)

        # 7. Reshape back to (B, T, F)
        z_q = z_q_flat.transpose(1,2) # (B, T, F)

        # 8. Select the disentangled representations
        spk_quantized = z_quantized[self.rvq_spk_idx].transpose(1,2) # (B, T, F)
        pros_quantized = z_quantized[self.rvq_pros_idx].transpose(1,2) # (B, T, F)
        text_quantized = z_quantized[-1].transpose(1,2) # (B, T, F)

        return z_q, spk_quantized, pros_quantized, text_quantized, commitment_loss, codebook_loss
import math
from typing import Dict, List, Tuple

from quick_convert.components.encoders.linguistic_head import LinguisticCTCHead
from quick_convert.components.encoders.pros_head import ProsodyHead
import torch
import torch.nn as nn
import torch.nn.functional as F

from quick_convert.components.encoders import (
    ParallelConformerEncoder, 
    SpeakerASPHead, 
    LinguisticCTCHead, 
    ProsodyHead
)

from quick_convert.components.layers import ResidualVectorQuantizer, GradientReversalLayer
from quick_convert.components.ssl.w2vbert import W2VBertContentEncoder

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
        linguistic_head: LinguisticCTCHead,
        speaker_head: SpeakerASPHead,
        prosody_head: ProsodyHead | None,
        emotion_head: ProsodyHead | None,
        rvq_idx : Dict[str, int] = {'content': 0, 'speaker': 1, 'prosody': 2, 'emotion': 2},
        *kwargs,
    ):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.content_encoder = content_encoder
        self.rvq = rvq

        self.speaker_head = speaker_head
        self.linguistic_head = linguistic_head
        self.prosody_head = prosody_head
        self.emotion_head = emotion_head

        if self.prosody_head is None and self.emotion_head is None:
            raise ValueError("At least one of prosody_head or emotion_head must be provided")

        self.rvq_idx = rvq_idx
        self._create_adversarial_heads()

    def _create_adversarial_heads(self):
        self.adv_speaker_head_ling = self.speaker_head.copy()  # For adversarial loss on speaker features
        self.adv_speaker_head_pros = self.speaker_head.copy()  # For adversarial loss on speaker features
        self.adv_linguistic_head_spk = self.linguistic_head.copy()  # For adversarial loss on linguistic features
        self.adv_linguistic_head_pros = self.linguistic_head.copy()  # For adversarial loss on linguistic features
        self.grl = GradientReversalLayer()  # For adversarial loss on speaker features

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
            waveform: torch.Tensor, 
            lengths: torch.Tensor
            ) -> List[torch.Tensor]:
        with torch.no_grad():
            return self.encode(waveform, lengths)[0:5]  # Return z_q, text_q, spk_q, pros_q, emo_q

    def encode(
        self,
        waveform: torch.Tensor,   # (B, T_samples)
        lengths: torch.Tensor,    # (B,)  — number of valid samples per item
    ) -> Tuple[
        torch.Tensor,             # z_q            (B, T, F)
        List[torch.Tensor],       # indices_list   [num_codebooks × (B·T,)]
        List[float],              # perplexity_list
        List[torch.Tensor],       # remainder_list (flat, length = num_codebooks + 1)
        torch.Tensor,             # spk_remainder  (B, T, F)
        torch.Tensor | None,      # pros_remainder (B, T, F)
        torch.Tensor | None,      # emo_remainder  (B, T, F)
    ]:
        """
        Args:
            waveform: Raw waveform tensor of shape (B, T_samples).
            lengths:  Number of valid waveform samples per batch item, shape (B,).

        Returns:
            z_q:             Differentiable RVQ reconstruction, shape (B, T, F).
            text_quantized:  Disentangled linguistic features, shape (B, T, F).
            spk_quantized:   Disentangled speaker features, shape (B, T, F).
            pros_quantized:  Disentangled prosody features, shape (B, T, F) or None if no prosody head.
            emo_quantized:   Disentangled emotion features, shape (B, T, F) or None if no emotion head.
            commitment_loss: Commitment loss from the RVQ, scalar tensor.
            codebook_loss:   Codebook loss from the RVQ, scalar tensor.
            lengths:         Updated lengths after feature extraction, shape (B,).
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
        text_quantized = z_quantized[self.rvq_idx['content']].transpose(1,2) # (B, T, F)
        spk_quantized = z_quantized[self.rvq_idx['speaker']].transpose(1,2) # (B, T, F)
        emo_pros_quantized = z_quantized[self.rvq_idx['prosody']].transpose(1,2) # (B, T, F)
        
        text_quantized = self.linguistic_head(text_quantized)
        spk_quantized = self.speaker_head(spk_quantized)
        
        pros_quantized = self.prosody_head(emo_pros_quantized) if self.prosody_head is not None else None
        emo_quantized = self.emotion_head(emo_pros_quantized) if self.emotion_head is not None else None

        # TODO - update lengths
        lengths = lengths

        return (
            z_q, 
            text_quantized, 
            spk_quantized, 
            pros_quantized, 
            emo_quantized, 
            commitment_loss, 
            codebook_loss, 
            lengths,
            z_quantized,
        )
    
    def compute_loss(
            self, 
            waveform: torch.FloatTensor, 
            lengths: torch.LongTensor,
            linguistic_targets: torch.LongTensor,
            target_lengths: torch.LongTensor,
            speaker_emb: torch.FloatTensor,
            emotion_seq: torch.FloatTensor | None,
            prosody_seq: torch.FloatTensor | None,
        ) -> List:

        z_q, text_q, spk_q, pros_q, emo_q, commitment_loss, codebook_loss, lengths, z_quantized = self.encode(waveform, lengths)

        rvq_losses = {
            'commitment_loss': commitment_loss,
            'codebook_loss': codebook_loss,
        }

        # Speaker loss: encourage spk_q to match the target speaker embedding
        spk_loss = self.speaker_head.compute_loss(spk_q, speaker_emb)

        # Prosody loss: encourage pros_q to match the target prosody features (if provided)
        if self.prosody_head is not None and prosody_seq is not None:
            pros_loss = self.prosody_head.compute_loss(pros_q, prosody_seq)
        else:
            pros_loss = 0.0

        # Emotion loss: encourage emo_q to match the target emotion features (if provided)
        if self.emotion_head is not None and emotion_seq is not None:
            emo_loss = self.emotion_head.compute_loss(emo_q, emotion_seq)
        else:            
            emo_loss = 0.0

        # CTC loss: encourage text_q to predict the target linguistic sequence
        ctc_loss = self.linguistic_head.compute_loss(
            text_q,
            linguistic_targets,
            input_lengths=lengths,
            target_lengths=target_lengths,
        )

        distill_losses = {
            'ctc_loss': ctc_loss,
            'spk_loss': spk_loss,
            'pros_loss': pros_loss,
            'emo_loss': emo_loss,
        }    

        # Adversarial speaker loss over linguistic features: encourage spk_q to be uninformative about speaker identity
        adv_spk_loss_ling = self.adv_speaker_head_ling.compute_loss(
            self.grl(z_quantized[self.rvq_idx['content']]), 
            speaker_emb
        )

        # Adversarial speaker loss over prosody features: encourage pros_q to be uninformative about speaker identity
        adv_spk_loss_pros = self.adv_speaker_head_pros.compute_loss(
            self.grl(z_quantized[self.rvq_idx['prosody']]), 
            speaker_emb
        )

        # Adversarial linguistic loss over speaker features: encourage text_q to be uninformative about linguistic content
        adv_ling_loss_spk = self.adv_linguistic_head_spk.compute_loss(
            self.grl(z_quantized[self.rvq_idx['speaker']]), 
            linguistic_targets,
            input_lengths=lengths,
            target_lengths=target_lengths,
        )

        # Adversarial linguistic loss over prosody features: encourage text_q to be uninformative about linguistic content
        adv_ling_loss_pros = self.adv_linguistic_head_pros.compute_loss(
            self.grl(z_quantized[self.rvq_idx['prosody']]), 
            linguistic_targets,
            input_lengths=lengths,
            target_lengths=target_lengths,
        )

        adv_losses = {
            'adv_spk_loss_ling': adv_spk_loss_ling,
            'adv_spk_loss_pros': adv_spk_loss_pros,
            'adv_ling_loss_spk': adv_ling_loss_spk,
            'adv_ling_loss_pros': adv_ling_loss_pros,
        }

        loss_dict = {
            'rvq_losses': rvq_losses,
            'distill_losses': distill_losses,
            'adv_losses': adv_losses,
        }
        
        return [
            z_q, 
            text_q, 
            spk_q, 
            pros_q, 
            emo_q, 
            loss_dict,
        ]

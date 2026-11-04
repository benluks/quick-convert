"""Reference-prefix conditioning for the first-party CosyVoice adapter."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from quick_convert.utils.masking import make_padding_mask


@dataclass(frozen=True)
class CosyVoiceReference:
    """Aligned content features and mels, with independent valid lengths.

    Features are [B, T, D]; mels are [B, M, mel_dim]. Neither includes source
    content. The decoder validates their frame-rate relationship before use.
    """

    features: torch.Tensor
    lengths: torch.Tensor
    mels: torch.Tensor
    mel_lengths: torch.Tensor

    def expand(self, batch_size, *, device, dtype):
        """Copy one prepared reference for a batch without exposing shared storage."""
        if self.features.shape[0] != 1 or self.mels.shape[0] != 1:
            raise ValueError("A prepared reference must contain exactly one recording.")
        return CosyVoiceReference(
            self.features.to(device=device, dtype=dtype).expand(batch_size, -1, -1).clone(),
            self.lengths.to(device=device).expand(batch_size).clone(),
            self.mels.to(device=device, dtype=dtype).expand(batch_size, -1, -1).clone(),
            self.mel_lengths.to(device=device).expand(batch_size).clone(),
        )

    def validate(self, batch_size: int, feature_dim: int, mel_dim: int, ratio: int) -> None:
        if self.features.ndim != 3 or self.features.shape[0] != batch_size or self.features.shape[2] != feature_dim:
            raise ValueError("Reference features must have shape [batch, time, feature_dim].")
        if self.mels.ndim != 3 or self.mels.shape[0] != batch_size or self.mels.shape[2] != mel_dim:
            raise ValueError("Reference mels must have shape [batch, time, mel_dim].")
        for values, lengths in ((self.features, self.lengths), (self.mels, self.mel_lengths)):
            if lengths.shape != (batch_size,) or lengths.dtype != torch.long:
                raise ValueError("Reference lengths must be integer vectors with one value per sample.")
            if torch.any(lengths <= 0) or torch.any(lengths > values.shape[1]):
                raise ValueError("Reference lengths exceed storage or contain empty sequences.")
            if not torch.isfinite(values).all():
                raise ValueError("Reference tensors must contain finite values.")
        if not torch.equal(self.lengths * ratio, self.mel_lengths):
            raise ValueError("Reference feature and mel lengths are not aligned.")


def reference_flow_loss(flow, features, lengths, mels, reference, speaker_embedding):
    """Train on compact reference+source sequences, scoring source frames only.

    This is the reference-prefix variant of the donor conditional flow-matching
    objective. The attention mask covers both sequences; the loss mask excludes
    the reference, so predicting the supplied prompt cannot lower the loss.
    """
    tokens = pad_sequence(
        [
            torch.cat((ref[:nr], src[:ns]))
            for ref, src, nr, ns in zip(reference.features, features, reference.lengths, lengths, strict=True)
        ],
        batch_first=True,
    )
    token_lengths = reference.lengths + lengths
    h, _ = flow.encoder(tokens, token_lengths, streaming=False)
    h = flow.encoder_proj(h)
    total_lengths = flow.output_lengths(token_lengths)
    if h.shape[1] < int(total_lengths.max()):
        raise ValueError("Flow encoder output is shorter than the reference and source.")
    h = h[:, : int(total_lengths.max())]
    targets = pad_sequence(
        [
            torch.cat((ref[:nr], src[:ns]))
            for ref, src, nr, ns in zip(
                reference.mels, mels, reference.mel_lengths, flow.output_lengths(lengths), strict=True
            )
        ],
        batch_first=True,
    ).transpose(1, 2)
    mask = make_padding_mask(total_lengths, max_length=h.shape[1])[:, None].to(h.dtype)
    positions = torch.arange(h.shape[1], device=h.device)[None]
    prompt_mask = positions < reference.mel_lengths[:, None]
    loss_mask = mask * (~prompt_mask[:, None])
    cond = targets * prompt_mask[:, None]
    spks = flow.spk_embed_affine_layer(F.normalize(speaker_embedding, dim=1))
    mu = h.transpose(1, 2)
    cfm = flow.decoder
    t = torch.rand((h.shape[0], 1, 1), device=h.device, dtype=h.dtype)
    noise = torch.randn_like(targets)
    y = (1 - (1 - cfm.sigma_min) * t) * noise + t * targets
    velocity = targets - (1 - cfm.sigma_min) * noise
    if cfm.training_cfg_rate > 0:
        keep = torch.rand(h.shape[0], device=h.device) > cfm.training_cfg_rate
        mu = mu * keep[:, None, None]
        spks = spks * keep[:, None]
        cond = cond * keep[:, None, None]
    prediction = cfm.estimator(y, mask, mu, t.flatten(), spks, cond, streaming=False)
    return ((prediction - velocity).square() * loss_mask).sum() / (loss_mask.sum() * targets.shape[1])

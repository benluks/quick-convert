"""Flow-matching operations for the adapter's named conditioning streams."""

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from quick_convert.utils.masking import make_padding_mask


def content_and_speaker(flow, features, lengths, speaker):
    """Prepare source-timed content without running auxiliary prefix values through it."""
    valid = make_padding_mask(lengths, max_length=features.shape[1])
    features = features.masked_fill(~valid[..., None], 0)
    if hasattr(flow, "encoder"):
        h, _ = flow.encoder(features, lengths, streaming=False)
        h = flow.encoder_proj(h)
    else:
        h = flow.pre_lookahead_layer(features)
        h = h.repeat_interleave(flow.token_mel_ratio, dim=1)
    mel_lengths = flow.output_lengths(lengths)
    h = h[:, : int(mel_lengths.max())]
    h = h.masked_fill(~make_padding_mask(mel_lengths, max_length=h.shape[1])[..., None], 0)
    spks = None if speaker is None else flow.spk_embed_affine_layer(F.normalize(speaker, dim=1))
    return h, spks


def stream_flow_loss(flow, h, spks, targets, lengths, conditions, loss_mask):
    """Score only the shared target span, keeping linguistic conditioning visible."""
    cfm = flow.decoder
    valid = make_padding_mask(lengths, max_length=targets.shape[1])[:, None]
    target = targets.transpose(1, 2)
    noise = torch.randn_like(target)
    t = torch.rand((target.shape[0], 1, 1), device=target.device, dtype=target.dtype)
    y = (1 - (1 - cfm.sigma_min) * t) * noise + t * target
    velocity = target - (1 - cfm.sigma_min) * noise
    keep = torch.rand(target.shape[0], device=target.device) >= cfm.training_cfg_rate
    conditions = conditions * keep[:, None, None]
    if spks is not None:
        spks = spks * keep[:, None]
    prediction = cfm.estimator(
        y, valid, h.transpose(1, 2), t.flatten(), spks, conditions.transpose(1, 2), streaming=False
    )
    score = (loss_mask & valid[:, 0])[:, None]
    return ((prediction - velocity).square().masked_fill(~score, 0)).sum() / (score.sum() * target.shape[1])


@torch.inference_mode()
def stream_flow_sample(flow, h, lengths, spks, prefixes, prefix_lengths, steps, cfg_rate):
    """Generate compact prefix+source sequences; return source-only padded mels."""
    cfm = flow.decoder
    h = pad_sequence(
        [F.pad(src[:n], (0, 0, int(p), 0)) for src, n, p in zip(h, lengths, prefix_lengths, strict=True)],
        batch_first=True,
    )
    total_lengths = prefix_lengths + lengths
    conditions = pad_sequence(
        [F.pad(src[:p], (0, 0, 0, int(n))) for src, p, n in zip(prefixes, prefix_lengths, lengths, strict=True)],
        batch_first=True,
    ).transpose(1, 2)
    mask = make_padding_mask(total_lengths, max_length=h.shape[1])[:, None]
    mu = h.transpose(1, 2)
    x = h.new_empty((h.shape[0], flow.output_size, h.shape[1])).normal_()
    times = torch.linspace(0, 1, steps + 1, device=h.device, dtype=h.dtype)
    if cfm.t_scheduler == "cosine":
        times = 1 - torch.cos(times * torch.pi / 2)
    elif cfm.t_scheduler != "linear":
        raise ValueError(f"Unsupported stream flow time scheduler: {cfm.t_scheduler}")
    for t, next_t in zip(times[:-1], times[1:], strict=True):
        velocity = cfm.estimator(x, mask, mu, t.expand(x.shape[0]), spks, conditions, streaming=False)
        if cfg_rate:
            # Null auxiliary context keeps the same source linguistic content.
            null = cfm.estimator(
                x,
                mask,
                mu,
                t.expand(x.shape[0]),
                None if spks is None else torch.zeros_like(spks),
                torch.zeros_like(conditions),
                streaming=False,
            )
            velocity = velocity + cfg_rate * (velocity - null)
        x = (x + (next_t - t) * velocity).masked_fill(~mask, 0)
    return pad_sequence(
        [sample[:, p : p + n].transpose(0, 1) for sample, p, n in zip(x, prefix_lengths, lengths, strict=True)],
        batch_first=True,
    ).transpose(1, 2)

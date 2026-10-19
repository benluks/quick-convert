"""Training and held-out scoring for experimental categorical duration probes."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from quick_convert.components.boundaries.duration import CategoricalDurationProbe


def probability_metrics(probabilities, labels, speakers):
    """Segment-weighted scores plus speaker-balanced NLL; lower is better."""
    loss = -probabilities[torch.arange(len(labels)), labels].clamp_min(torch.finfo(probabilities.dtype).tiny).log()
    targets = F.one_hot(labels, num_classes=probabilities.shape[1]).float()
    confidence, predicted = probabilities.max(-1)
    correct = predicted == labels
    ece = 0.0
    for i in range(10):
        mask = (confidence >= i / 10) & ((confidence < (i + 1) / 10) if i < 9 else (confidence <= 1))
        if mask.any():
            ece += float(mask.float().mean() * (correct[mask].float().mean() - confidence[mask].mean()).abs())
    speaker_nll = {
        speaker: float(loss[torch.tensor([s == speaker for s in speakers])].mean()) for speaker in sorted(set(speakers))
    }
    return dict(
        count=len(labels),
        nll=float(loss.mean()),
        brier=float((probabilities - targets).square().sum(-1).mean()),
        accuracy=float(correct.float().mean()),
        ece_10_bins=ece,
        speaker_macro_nll=sum(speaker_nll.values()) / len(speaker_nll),
        by_speaker_nll=speaker_nll,
    )


def macro_gain_interval(baseline, model, *, seed=115, repetitions=1000):
    """Paired speaker bootstrap of baseline-minus-model mean NLL."""
    speakers = sorted(model["by_speaker_nll"])
    if len(speakers) < 2:
        return None
    gains = torch.tensor(
        [baseline["by_speaker_nll"][s] - model["by_speaker_nll"][s] for s in speakers], dtype=torch.float64
    )
    indices = torch.randint(len(gains), (repetitions, len(gains)), generator=torch.Generator().manual_seed(seed))
    return torch.quantile(gains[indices].mean(1), torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()


@torch.no_grad()
def predict_probabilities(model, features, device, batch_size=1024):
    model.eval()
    return torch.cat([model(chunk.to(device)).softmax(-1).cpu() for chunk in features.split(batch_size)])


def fit_duration_probe(
    features,
    labels,
    prior,
    *,
    hidden_dim=128,
    dropout=0.1,
    max_steps=1000,
    batch_size=256,
    learning_rate=1e-3,
    weight_decay=1e-3,
    eval_interval=100,
    patience=5,
    device="cpu",
    seed=115,
):
    """Fit normalization on train, choose checkpoint by validation NLL only.

    Test features/labels are not used during fitting or checkpoint selection.
    Features and labels mappings need contain only train and validation entries.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    center = features["train"].mean(0)
    scale = features["train"].std(0, correction=0).clamp_min(1e-5)
    train = (features["train"] - center) / scale
    validation = (features["validation"] - center) / scale
    model = CategoricalDurationProbe(train.shape[1], len(prior), hidden_dim, dropout).to(device)
    # Step zero is exactly the training-frequency baseline.
    torch.nn.init.zeros_(model.network[-1].weight)
    with torch.no_grad():
        model.network[-1].bias.copy_(prior.log().to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    initial = predict_probabilities(model, validation, device)
    best_loss = float(F.nll_loss(initial.clamp_min(torch.finfo(initial.dtype).tiny).log(), labels["validation"]))
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_step = bad_checks = 0
    history = [dict(step=0, validation_nll=best_loss)]
    for step in range(1, max_steps + 1):
        model.train()
        index = torch.randint(len(train), (min(batch_size, len(train)),), generator=generator)
        logits = model(train[index].to(device))
        loss = F.cross_entropy(logits, labels["train"][index].to(device))
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite categorical probe training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % eval_interval and step != max_steps:
            continue
        prediction = predict_probabilities(model, validation, device)
        val_loss = float(
            F.nll_loss(prediction.clamp_min(torch.finfo(prediction.dtype).tiny).log(), labels["validation"])
        )
        history.append(dict(step=step, validation_nll=val_loss))
        if val_loss < best_loss - 1e-6:
            best_loss, best_step, bad_checks = val_loss, step, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad_checks += 1
        if bad_checks >= patience:
            break
    model.load_state_dict(best_state)
    model.to(device).eval()
    return model, center, scale, dict(best_step=best_step, steps_run=step, validation_history=history)


def fit_frame_distribution(training_frames, edges):
    """Empirical within-bin frame distributions, using training durations only."""
    from quick_convert.components.boundaries.duration import duration_labels

    support, counts = training_frames.unique(sorted=True, return_counts=True)
    bins = duration_labels(support, edges)
    totals = torch.zeros(len(edges) + 1).scatter_add_(0, bins, counts.float())
    if (totals == 0).any():
        raise ValueError("Every duration bin needs training examples")
    return dict(support_frames=support, support_bins=bins, conditional_mass=counts.float() / totals[bins])


def duration_error_metrics(probabilities, frames, edges, distribution, frame_hz):
    """Modal-bin interval error, mixture-median MAE, and normalized ordinal RPS.

    Last-bin interval is unbounded above. Point estimates use the median of the
    probability-weighted training within-bin distributions, including the tail.
    RPS uses bin order, not unequal physical bin widths. No fitting occurs here.
    """
    from quick_convert.components.boundaries.duration import duration_labels

    if not math.isfinite(frame_hz) or frame_hz <= 0:
        raise ValueError("frame_hz must be positive and finite")
    frames, edges, probabilities = frames.cpu(), edges.cpu(), probabilities.cpu()
    labels = duration_labels(frames, edges)
    predicted = probabilities.argmax(-1)
    lower = torch.cat((torch.ones(1, dtype=edges.dtype), edges + 1))[predicted]
    upper = torch.cat((edges.double(), torch.tensor([float("inf")])))[predicted]
    interval = (lower - frames).clamp_min(0) + (frames - upper).clamp_min(0)
    support = distribution["support_frames"].cpu()
    bins = distribution["support_bins"].cpu()
    mass = distribution["conditional_mass"].cpu()
    estimates = []
    for chunk in probabilities.split(256):
        cdf = (chunk[:, bins] * mass).cumsum(-1)
        # First support point whose cumulative probability reaches one half.
        indices = (cdf < 0.5).sum(-1).clamp_max(len(support) - 1)
        estimates.append(support[indices])
    absolute = (torch.cat(estimates) - frames).abs().double()
    if probabilities.shape[1] > 1:
        thresholds = torch.arange(probabilities.shape[1] - 1)
        rps = (probabilities.cumsum(-1)[:, :-1] - (labels[:, None] <= thresholds).float()).square().mean()
    else:
        rps = torch.tensor(0.0)
    return dict(
        modal_bin_interval_mae_frames=float(interval.mean()),
        modal_bin_interval_mae_ms=float(interval.mean()) * 1000 / frame_hz,
        mixture_median_mae_frames=float(absolute.mean()),
        mixture_median_mae_ms=float(absolute.mean()) * 1000 / frame_hz,
        mixture_median_median_absolute_error_frames=float(absolute.quantile(0.5)),
        mixture_median_median_absolute_error_ms=float(absolute.quantile(0.5)) * 1000 / frame_hz,
        ranked_probability_score=float(rps),
    )

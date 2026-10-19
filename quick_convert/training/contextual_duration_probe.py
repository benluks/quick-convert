"""Matched independent/contextual training on ordered utterance sequences."""

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from quick_convert.components.boundaries.duration import CategoricalDurationProbe, ContextualDurationCNN


def sequence_batch(sequences, device):
    """Pad segment-vector sequences and return an explicit validity mask."""
    lengths = torch.tensor([len(s) for s in sequences], device=device)
    features = pad_sequence(sequences, batch_first=True).to(device)
    mask = torch.arange(features.shape[1], device=device)[None] < lengths[:, None]
    return features, mask


def sequence_logits(model, features, mask):
    return model(features, mask) if isinstance(model, ContextualDurationCNN) else model(features)


@torch.no_grad()
def predict_sequences(model, sequences, center, scale, device, batch_size=8):
    """Flatten only valid predictions in utterance/segment order."""
    model.eval()
    results = []
    for start in range(0, len(sequences), batch_size):
        batch = [(x - center) / scale for x in sequences[start : start + batch_size]]
        features, mask = sequence_batch(batch, device)
        results.append(sequence_logits(model, features, mask)[mask].softmax(-1).cpu())
    return torch.cat(results)


def fit_sequence_probe(
    features,
    labels,
    prior,
    *,
    architecture,
    hidden_dim=128,
    dropout=0.1,
    max_steps=1000,
    batch_size=8,
    learning_rate=1e-3,
    weight_decay=1e-3,
    eval_interval=100,
    patience=5,
    device="cpu",
    seed=115,
):
    """Train on utterances; select by segment-weighted validation NLL only.

    Both architectures use identical batches, targets, normalization and bins.
    Each sampled utterance contributes all valid segments; padding has no loss.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    train_flat = torch.cat(features["train"])
    center = train_flat.mean(0)
    scale = train_flat.std(0, correction=0).clamp_min(1e-5)
    constructor = {"independent": CategoricalDurationProbe, "cnn": ContextualDurationCNN}[architecture]
    model = constructor(len(center), len(prior), hidden_dim, dropout).to(device)
    head = model.output if architecture == "cnn" else model.network[-1]
    torch.nn.init.zeros_(head.weight)
    with torch.no_grad():
        head.bias.copy_(prior.log().to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    validation_labels = torch.cat(labels["validation"])

    def validation_loss():
        probs = predict_sequences(model, features["validation"], center, scale, device, batch_size)
        return float(F.nll_loss(probs.clamp_min(torch.finfo(probs.dtype).tiny).log(), validation_labels))

    best_loss = validation_loss()
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_step = bad_checks = 0
    history = [dict(step=0, validation_nll=best_loss)]
    for step in range(1, max_steps + 1):
        model.train()
        indices = torch.randint(len(features["train"]), (batch_size,), generator=generator).tolist()
        batch = [(features["train"][i] - center) / scale for i in indices]
        x, mask = sequence_batch(batch, device)
        y = pad_sequence([labels["train"][i] for i in indices], batch_first=True).to(device)
        loss = F.cross_entropy(sequence_logits(model, x, mask)[mask], y[mask])
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite sequence probe training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % eval_interval and step != max_steps:
            continue
        val_loss = validation_loss()
        history.append(dict(step=step, validation_nll=val_loss))
        if val_loss < best_loss - 1e-6:
            best_loss, best_step, bad_checks = val_loss, step, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_checks += 1
        if bad_checks >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, center, scale, dict(best_step=best_step, steps_run=step, validation_history=history)

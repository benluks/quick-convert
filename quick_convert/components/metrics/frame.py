"""Streaming metrics for masked binary, ordinal-bin, and continuous predictions."""

import torch


class FramePredictionMetrics:
    """Accumulate frame statistics independently of losses, models, and loggers.

    ``update`` honors the aligned HeadTarget validity mask. ``merge`` combines
    compatible local accumulators. ``compute(sync_dist=True)`` reduces a copy
    of the statistics, so repeated computation never double-counts workers.
    State is ephemeral and is not registered as model/checkpoint state."""

    def __init__(self, kind, classes, device):
        if kind not in {"bins", "binary", "regression"}:
            raise ValueError("Metric kind must be bins, binary, or regression.")
        if type(classes) is not int:
            raise ValueError("classes must be an integer.")
        if (
            (kind == "bins" and classes < 2)
            or (kind == "binary" and classes != 2)
            or (kind == "regression" and classes != 0)
        ):
            raise ValueError("Use at least two bin classes, two binary classes, or zero regression classes.")
        self.kind = kind
        # count, absolute/squared error, correct, TP, predicted/target positive,
        # followed by prediction and target histograms for categorical outputs.
        self.classes = classes
        self.state = torch.zeros(
            7 + 3 * classes, device=device, dtype=torch.float32 if torch.device(device).type == "mps" else torch.float64
        )

    @torch.no_grad()
    def update(self, predictions, target, padding_mask):
        if predictions.device != self.state.device:
            raise ValueError("Predictions must be on the metric's device.")
        mask = target.valid_mask(padding_mask).to(predictions.device)
        if predictions.shape[:2] != mask.shape or predictions.ndim not in (2, 3):
            raise ValueError("Predictions must match the aligned batch/time dimensions.")
        if self.kind == "bins":
            if predictions.ndim != 3 or predictions.shape[-1] != self.classes or target.values.ndim != 2:
                raise ValueError("Binned predictions require [B,T,classes] logits and [B,T] targets.")
        elif (predictions.ndim == 3 and predictions.shape[-1] != 1) or (
            target.values.ndim not in (2, 3) or (target.values.ndim == 3 and target.values.shape[-1] != 1)
        ):
            raise ValueError("Binary/regression predictions and targets must be [B,T] or [B,T,1].")
        values = target.values.to(predictions.device)[mask].detach()
        logits = predictions[mask].detach()
        if not torch.isfinite(values).all() or not torch.isfinite(logits).all():
            raise ValueError("Observed predictions and targets must be finite.")
        if self.kind == "bins" and (
            values.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
            or torch.any(values < 0)
            or torch.any(values >= self.classes)
        ):
            raise ValueError("Binned targets must be integer class indices within range.")
        if self.kind == "binary" and torch.any((values != 0) & (values != 1)):
            raise ValueError("Binary targets must be zero or one.")
        if self.kind == "bins":
            predicted = logits.argmax(-1)
            values = values.long()
        elif self.kind == "binary":
            predicted = (logits.reshape(-1) >= 0).long()
            values = values.reshape(-1).long()
        else:
            predicted = logits.reshape(-1)
            values = values.reshape(-1)
        error = predicted.to(self.state.dtype) - values.to(self.state.dtype)
        self.state[:4] += torch.stack(
            (error.new_tensor(values.numel()), error.abs().sum(), error.square().sum(), (predicted == values).sum())
        )
        if self.kind == "binary":
            self.state[4:7] += torch.stack(((predicted * values).sum(), predicted.sum(), values.sum()))
        if self.classes:
            self.state[7 : 7 + self.classes] += torch.bincount(predicted, minlength=self.classes)
            self.state[7 + self.classes : 7 + 2 * self.classes] += torch.bincount(values, minlength=self.classes)
            self.state[7 + 2 * self.classes :] += torch.bincount(values[predicted == values], minlength=self.classes)

    def reset(self):
        """Start an empty accumulation on the same device."""
        self.state.zero_()

    def merge(self, other):
        """Add another compatible accumulator without mutating it."""
        if self.kind != other.kind or self.classes != other.classes:
            raise ValueError("Cannot merge metrics with different prediction contracts.")
        self.state += other.state.to(self.state)

    def compute(self, *, sync_dist=False):
        """Compute ratios from totals; optionally sum a snapshot across workers."""
        state = self.state.clone()
        if sync_dist and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(state)
        count, absolute, squared, correct, tp, predicted_positive, target_positive = state[:7]
        result = {"valid_frames": count}
        if count == 0:
            return result
        result["mae_bins" if self.kind == "bins" else "mae"] = absolute / count
        if self.kind == "regression":
            result["rmse"] = (squared / count).sqrt()
        else:
            result["accuracy"] = correct / count
            target_counts = state[7 + self.classes : 7 + 2 * self.classes]
            result["majority_baseline_accuracy"] = target_counts.max() / count
            for index in range(self.classes):
                result[f"predicted_fraction/bin_{index}"] = state[7 + index] / count
                result[f"target_fraction/bin_{index}"] = target_counts[index] / count
                if target_counts[index] > 0:
                    result[f"recall/bin_{index}"] = state[7 + 2 * self.classes + index] / target_counts[index]
        if self.kind == "binary":
            result.update(
                precision=tp / predicted_positive.clamp_min(1),
                recall=tp / target_positive.clamp_min(1),
                f1=2 * tp / (predicted_positive + target_positive).clamp_min(1),
                predicted_voiced_fraction=predicted_positive / count,
                target_voiced_fraction=target_positive / count,
            )
        return result

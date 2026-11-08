# Frame prediction diagnostics

`FramePredictionMetrics` accumulates masked statistics for already aligned
frame predictions. It is independent of systems, losses and training frameworks.
Use it explicitly in a custom evaluation loop or training module; existing
training recipes do not automatically add these metrics.

```python
import torch
from quick_convert.components.heads.base import HeadTarget
from quick_convert.components.metrics import FramePredictionMetrics

target = HeadTarget(
    values=torch.tensor([[0, 1, 0]]),
    lengths=torch.tensor([3]),
    observation_mask=torch.tensor([[True, True, False]]),
)
metric = FramePredictionMetrics("bins", classes=2, device="cpu")
logits = torch.tensor([[[3.0, 0.0], [0.0, 3.0], [2.0, 1.0]]])
metric.update(logits, target, torch.ones(1, 3, dtype=torch.bool))
scores = metric.compute()
metric.reset()
```

`HeadTarget.valid_mask` combines prediction validity, target lengths and optional
observation validity. True means usable. Unobserved targets may contain sentinel
values or NaNs; observed values must be valid. Matching tensor shapes do not
establish temporal alignment. Align the time grids before computing statistics.

`bins` accepts `[B,T,C]` logits and integer `[B,T]` targets. It reports accuracy,
ordinal MAE in bin indices, prediction/target fractions, per-class recall and the
majority-class baseline. Bin-index MAE is not a physical-unit error.
`binary` accepts logits and zero/one targets; zero logits count as positive. It
reports precision/recall/F1 and positive-frame fractions. `regression` reports
MAE/RMSE in the target's supplied units. Binary/regression values may be `[B,T]`
or `[B,T,1]`. Empty accumulations return only `valid_frames`.

`merge` combines compatible local accumulators. `compute(sync_dist=True)` sums
a snapshot across distributed workers without changing local state; every rank
must participate with the same metric collections. Metric counters are ephemeral
and are not added to model checkpoints. Predictions must use the metric's device.

The existing `MediaLogger` supports `log_frame_predictions` for one example's
target/prediction contour and `log_table` for per-bin diagnostics. Optional
`HeadTarget.timestamps` supply aligned `[B,T]` times in seconds; otherwise contours
use frame indices. Contours exclude invalid frames. W&B receives native tables;
text-capable backends receive Markdown tables. The null logger skips rendering.
Matplotlib and logger backends remain optional dependencies of training.

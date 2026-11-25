"""Measure real training throughput without changing the training loop."""

import json
import math
from pathlib import Path
from time import perf_counter

import lightning as L
import torch


class TrainingThroughput(L.Callback):
    """Synchronize CUDA at batch boundaries for explicit timing measurements."""

    def __init__(self, warmup_batches: int = 10, output: str = "throughput.json"):
        if warmup_batches < 0:
            raise ValueError("warmup_batches must be nonnegative.")
        self.warmup_batches = warmup_batches
        self.output = output
        self.reports = []
        self.seen = 0

    def on_fit_start(self, trainer, pl_module):
        if trainer.world_size != 1:
            raise ValueError("TrainingThroughput currently supports single-device training only.")
        self.device = pl_module.device
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("TrainingThroughput supports CPU and CUDA.")
        self.reports = []
        self.seen = 0

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def on_train_epoch_start(self, trainer, pl_module):
        self._sync()
        self.epoch_start = self.previous_end = perf_counter()
        self.times = []
        self.gaps = []
        self.audio_seconds = 0.0
        self.examples = 0
        self.window_start = None
        self.window_end = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._sync()
        self.measuring = self.seen >= self.warmup_batches
        if not self.measuring:
            return
        if batch.lengths is None or batch.sample_rates is None:
            raise ValueError("TrainingThroughput requires loaded audio lengths and sample rates.")
        self.audio_seconds += (batch.lengths.double() / batch.sample_rates).sum().item()
        self.examples += len(batch)
        if self.window_start is None:
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            self.window_start = perf_counter()
        self.batch_start = perf_counter()
        if self.times:
            self.gaps.append(self.batch_start - self.previous_end)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._sync()
        now = perf_counter()
        self.seen += 1
        if self.measuring:
            self.times.append(now - self.batch_start)
            self.window_end = now
        self.previous_end = now

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.times:
            return
        elapsed = self.window_end - self.window_start
        ordered = sorted(self.times)
        report = {
            "epoch": trainer.current_epoch,
            "batches": len(self.times),
            "examples": self.examples,
            "audio_seconds": self.audio_seconds,
            "measured_seconds": elapsed,
            "audio_seconds_per_second": self.audio_seconds / elapsed,
            "examples_per_second": self.examples / elapsed,
            "batch_p50_seconds": ordered[math.ceil(len(ordered) * 0.5) - 1],
            "batch_p95_seconds": ordered[math.ceil(len(ordered) * 0.95) - 1],
            "inter_batch_seconds": sum(self.gaps),
            "epoch_hook_seconds": perf_counter() - self.epoch_start,
        }
        if self.device.type == "cuda":
            report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(self.device)
            report["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(self.device)
        self.reports.append(report)
        path = Path(trainer.default_root_dir) / self.output
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.reports, indent=2) + "\n")
        print(json.dumps(report, indent=2))

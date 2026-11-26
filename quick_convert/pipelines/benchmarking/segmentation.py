"""Compare segmentation and batching without decoding waveforms."""

import json
from pathlib import Path

from quick_convert.data.base_dataset import BaseDataset
from quick_convert.data.segmentation import FixedDurationSegmentation

from .batching import BatchingBenchmarkPipeline


class SegmentationBenchmarkPipeline:
    def __init__(self, dataset, strategies, durations, seeds=(0,), duration_cache=None, output=None):
        self.dataset = dataset
        self.strategies = strategies
        self.durations = durations
        self.seeds = seeds
        self.duration_cache = duration_cache
        self.output = output

    def run(self):
        reports = {}
        for duration in [None, *self.durations]:
            name = "original" if duration is None else f"{duration:g}s"
            print(f"\nSegmentation: {name}")
            dataset = self.dataset
            if duration is not None:
                if dataset.resource_providers or dataset.max_length is not None:
                    raise ValueError("Segmentation benchmark requires variable-length audio without providers.")
                dataset = BaseDataset(
                    rows=dataset.rows,
                    segmentation=FixedDurationSegmentation(duration, duration_cache=self.duration_cache),
                )
            report = BatchingBenchmarkPipeline(
                dataset, self.strategies, seeds=self.seeds, duration_cache=self.duration_cache
            ).run()
            baseline = reports.get("original", report)
            report["audio_retained_fraction"] = report["duration_seconds"] / baseline["duration_seconds"]
            report["intrinsic_attention_ratio"] = (
                report["intrinsic_attention_cost"] / baseline["intrinsic_attention_cost"]
            )
            reports[name] = report
        print(f"{'Segmentation':>14} {'Examples':>10} {'Retained':>10} {'Intrinsic attention':>20}")
        for name, report in reports.items():
            print(
                f"{name:>14} {report['examples']:10} {report['audio_retained_fraction']:10.2%} "
                f"{report['intrinsic_attention_ratio']:20.2%}"
            )
        if self.output is not None:
            path = Path(self.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(reports, indent=2) + "\n")
        return reports

"""Compare batch schedules without decoding audio."""

import json
from pathlib import Path

import torch

from quick_convert.data.durations import resolve_durations
from quick_convert.data.sampling import LengthAwareBatchSampler, batching_statistics


class BatchingBenchmarkPipeline:
    def __init__(self, dataset, strategies, seeds=(0,), duration_cache=None, output=None):
        self.dataset = dataset
        self.strategies = strategies
        self.seeds = seeds
        self.duration_cache = duration_cache
        self.output = output

    def run(self):
        if self.dataset.max_length is not None:
            raise ValueError("Benchmark requires variable-length collation.")
        lengths = resolve_durations(self.dataset.rows, cache_path=self.duration_cache)
        if not lengths:
            raise ValueError("Cannot benchmark an empty dataset.")
        results = []
        for seed in self.seeds:
            for name, options in self.strategies.items():
                options = dict(options)
                kind = options.pop("kind")
                if kind == "random":
                    batch_size = options.pop("batch_size")
                    if batch_size < 1 or options:
                        raise ValueError("Invalid random batching options.")
                    indices = torch.randperm(len(lengths), generator=torch.Generator().manual_seed(seed)).tolist()
                    batches = [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]
                elif kind == "length_aware":
                    batches = list(LengthAwareBatchSampler(lengths, seed=seed, **options))
                else:
                    raise ValueError(f"Unknown batching strategy: {kind}")
                if sorted(i for batch in batches for i in batch) != list(range(len(lengths))):
                    raise ValueError(f"Strategy {name} does not cover every example exactly once.")
                stats = batching_statistics(lengths, batches)
                stats.update(
                    name=name,
                    seed=seed,
                    min_batch_size=min(map(len, batches)),
                    max_batch_size=max(map(len, batches)),
                    mean_batch_size=len(lengths) / len(batches),
                    max_padded_batch_seconds=max(len(batch) * max(lengths[i] for i in batch) for batch in batches),
                )
                results.append(stats)
        print(f"{'Strategy':24} {'Seed':>5} {'Batches':>8} {'Padding':>9} {'Padded (h)':>12} {'Attention':>14}")
        for row in results:
            print(
                f"{row['name']:24} {row['seed']:5} {row['batches']:8} "
                f"{row['padding_fraction']:9.1%} {row['padded_volume'] / 3600:12.2f} "
                f"{row['attention_cost']:14.0f}"
            )
        report = {"examples": len(lengths), "duration_seconds": sum(lengths), "results": results}
        if self.output is not None:
            path = Path(self.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2) + "\n")
        return report

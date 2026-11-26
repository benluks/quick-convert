"""Compare batch schedules without decoding audio."""

import json
import math
from pathlib import Path

import torch

from quick_convert.data.durations import resolve_durations
from quick_convert.data.sampling import LengthAwareBatchSampler, batching_statistics


def cost_percentiles(values):
    """Nearest-rank percentiles, including the observed maximum."""
    ordered = sorted(values)
    return {f"p{p}": ordered[math.ceil(len(ordered) * p / 100) - 1] for p in (50, 90, 95, 99, 100)}


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
        intrinsic_attention = sum(length**2 for length in lengths)
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
                linear_costs = [len(batch) * max(lengths[i] for i in batch) for batch in batches]
                quadratic_costs = [len(batch) * max(lengths[i] for i in batch) ** 2 for batch in batches]
                stats.update(
                    name=name,
                    distinct_batch_shapes=len(
                        {(len(batch), round(max(lengths[i] for i in batch) * 16000)) for batch in batches}
                    ),
                    seed=seed,
                    min_batch_size=min(map(len, batches)),
                    max_batch_size=max(map(len, batches)),
                    mean_batch_size=len(lengths) / len(batches),
                    max_padded_batch_seconds=max(linear_costs),
                    batch_linear_cost=cost_percentiles(linear_costs),
                    batch_attention_cost=cost_percentiles(quadratic_costs),
                    attention_padding_fraction=1 - intrinsic_attention / stats["attention_cost"],
                    strategy_options=dict(self.strategies[name]),
                )
                results.append(stats)
        print(f"{'Strategy':24} {'Seed':>5} {'Batches':>8} {'Padding':>9} {'Padded (h)':>12} {'Attention':>14}")
        for row in results:
            print(
                f"{row['name']:24} {row['seed']:5} {row['batches']:8} "
                f"{row['padding_fraction']:9.1%} {row['padded_volume'] / 3600:12.2f} "
                f"{row['attention_cost']:14.0f}"
            )
        print(f"Intrinsic unpadded attention cost: {intrinsic_attention:.0f} seconds squared")
        print(f"{'Strategy':24} {'Seed':>5} {'Peak (s)':>10} {'P95 (s)':>10} {'Peak attn':>12} {'P95 attn':>12}")
        for row in results:
            print(
                f"{row['name']:24} {row['seed']:5} {row['batch_linear_cost']['p100']:10.1f} "
                f"{row['batch_linear_cost']['p95']:10.1f} {row['batch_attention_cost']['p100']:12.0f} "
                f"{row['batch_attention_cost']['p95']:12.0f}"
            )
        report = {
            "examples": len(lengths),
            "duration_seconds": sum(lengths),
            "intrinsic_attention_cost": intrinsic_attention,
            "results": results,
        }
        if self.output is not None:
            path = Path(self.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2) + "\n")
        return report

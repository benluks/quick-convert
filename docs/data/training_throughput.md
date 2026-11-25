# Training throughput

Append this callback to the existing Lightning trainer callbacks (retain any
checkpointing and other callbacks):

```yaml
trainer:
  trainer_kwargs:
    callbacks:
      - _target_: quick_convert.training.lightning.throughput.TrainingThroughput
        warmup_batches: 10
        output: throughput.json
```

The callback writes a JSON list of epoch reports under the Lightning default
root directory (or to an absolute output path). It measures loaded source audio
duration from unpadded lengths and sample rates. It supports CPU and single CUDA
device training; metadata-only batches and distributed training are unsupported.

Warmup batches are excluded once per fit, not once per epoch. The measured
window spans the first measured batch start through the last measured batch
end in each epoch. Inter-batch time includes loading, transfer, callbacks and
other loop overhead; it is not isolated dataloader latency. Epoch hook duration
can include validation and other hooks. CUDA synchronization at batch boundaries
makes timing explicit but can perturb throughput. Peak CUDA allocator statistics
are reset at the start of each measured window and include live allocations.

Compare fresh runs with identical seeds, dataset coverage, precision, model,
logging and validation settings. Disable mid-epoch validation for steady-state
training comparisons. Use unique output directories and compare measured audio
seconds per wall-clock second, not just batches per second. Different batch sizes
change optimizer update counts and may change convergence. These measurements
are not a convergence benchmark.

# Configuration reference

See [optional dependency planning](dependencies.md) for deriving installation
requirements from a composed run.

Quick Convert installs its Hydra tree under `quick_convert/configs/`. A file in `run/` is an executable composition root rather than a separate kind of Python object.

## CLI resolution

| Invocation | Selected run config |
|---|---|
| `quick-convert train_vq_asr_librispeech` | `run/train_vq_asr_librispeech.yaml` |
| `train vq_asr_librispeech` | `run/train_vq_asr_librispeech.yaml` |
| `precompute tokens_librispeech` | `run/precompute_tokens_librispeech.yaml` |
| `evaluate asr_librispeech` | `run/eval_asr_librispeech.yaml` |
| `anonymize knnvc_clac` | `run/anonymize_knnvc_clac.yaml` |

The universal command uses the full stem. A verb command prefixes its alias with `train_`, `precompute_`, `eval_`, or `anonymize_`.

## Composition slots

Run configs commonly select objects into named slots:

- `pipeline`: workflow orchestration;
- `system`: inference-ready task object;
- `trainer`: training backend and module;
- `source_dataset`: dataset used to derive or precompute resources;
- `train_dataset` and `val_dataset`: manifest-backed training inputs;
- `feature_extractor`: component used by precomputation.

The slot name matters because interpolations and instantiated constructor arguments refer to it. Prefer explicit names over deeply nested generic containers.

## Overrides

Hydra overrides use dotted paths:

```bash
uv run train vq_asr_librispeech \
    data_root=/data \
    out_root=/scratch/quick-convert \
    device=cuda \
    trainer.train_dataloader_kwargs.batch_size=16
```

Quote shell-sensitive lists and strings. Use `HYDRA_FULL_ERROR=1` when debugging a composition or instantiation failure.

## Learning-rate duration

The `adamw_cosine` optimizer preset uses `T_max: auto`. Training supplies
Lightning's estimated optimizer-step budget, accounting for accumulation and
the configured training limits. Automatic cosine decay uses the remaining
steps after warmup and stays at `eta_min` after its duration ends.

```yaml
trainer:
  module:
    optimization:
      warmup:
        steps: 0.05
      lr_scheduler_kwargs:
        T_max: auto
        eta_min: 1e-6
```

An integer warmup duration means an absolute number of steps; a floating-point
duration in `(0, 1]` means a fraction of total optimizer steps. Automatic cosine
requires a finite positive training budget, step-based scheduling with frequency
one, and enough steps remaining after warmup. Numeric `T_max` overrides retain
PyTorch's existing cosine behavior. Resume restores the saved scheduler duration;
extending the run does not automatically stretch an existing checkpoint's schedule.


## Adding a run

1. Reuse existing config groups where possible.
2. Put the composition root in `quick_convert/configs/run/`.
3. Use a verb prefix so the corresponding CLI alias can discover it.
4. Keep the task model at top-level `system` when the workflow has one.
5. Add the run to tests by relying on the existing all-run composition check.
6. Confirm the run is present in an installed wheel and in `quick-convert --help`.

See [Hydra structure](hydra_structure.md) for a guided composition example.

## Trainable online encoders

SSL reconstruction and VQ-ASR systems freeze online encoders by default. Select
roles explicitly to allow downstream losses to update them:

```yaml
system:
  trainable_online_encoders: [content]
```

For an existing reconstruction run, use the Hydra override
`'+system.trainable_online_encoders=[content]'`. WavLM and W2V-BERT support
autograd through their online forwards. Other backends must also support
autograd; selecting a role cannot bypass a backend's internal inference context.
Precomputed batch resources take precedence and remain detached, so omit cached
content resources when fine-tuning the online content encoder. Encoder parameters
are registered on the system and included by the training module's optimizer.

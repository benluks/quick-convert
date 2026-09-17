# Configuration reference

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

## Adding a run

1. Reuse existing config groups where possible.
2. Put the composition root in `quick_convert/configs/run/`.
3. Use a verb prefix so the corresponding CLI alias can discover it.
4. Keep the task model at top-level `system` when the workflow has one.
5. Add the run to tests by relying on the existing all-run composition check.
6. Confirm the run is present in an installed wheel and in `quick-convert --help`.

See [Hydra structure](hydra_structure.md) for a guided composition example.

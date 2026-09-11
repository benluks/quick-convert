## Installation

Quick Convert uses [uv](https://docs.astral.sh/uv/) for Python environment
management:

```bash
git clone https://github.com/benluks/quick-convert.git
cd quick-convert
uv sync
```

## Running pipelines

Pipeline-style files in `configs/run/` can be launched through the universal
entrypoint using the complete filename stem:

```bash
uv run quick-convert train_vq_asr_librispeech pipeline.batch_size=16
```

Common operations also have shorter, verb-oriented aliases:

```bash
uv run train vq_asr_librispeech pipeline.batch_size=16
uv run evaluate asr_librispeech
uv run anonymize knnvc_clac target_id=6081
uv run precompute content_w2vbert_librispeech
uv run build_manifest libri
```

An alias prepends its config prefix. For example, `evaluate asr_librispeech`
selects `configs/run/eval_asr_librispeech.yaml`; `train vq_asr_librispeech`
selects `configs/run/train_vq_asr_librispeech.yaml`. The older
`eval_asr librispeech` command remains available for compatibility, but new
usage should prefer `evaluate`.

Arguments after the config name are passed to Hydra as overrides. Run a command
without a config name to list the configurations it can resolve:

```bash
uv run evaluate
uv run quick-convert
```

The shared runner composes the selected config, instantiates `cfg.pipeline`,
passes the resolved config to `write_config()` when the pipeline provides that
method, and calls `pipeline.run(**cfg.run)`. This keeps the CLI orchestration
generic while allowing pipelines to expose their own run arguments.

Special-purpose utilities whose interfaces do not follow this pipeline
contract, including ASV evaluation and manifest splitting, remain available as
Python modules under `quick_convert.cli`.

# Installation

Quick Convert requires Python 3.11 or newer. The base installation provides data handling, core model interfaces, Hydra configuration, and the universal CLI.

## Development checkout

```bash
git clone https://github.com/benluks/quick-convert.git
cd quick-convert
uv sync --group dev
```

Run `uv run quick-convert --help` to confirm that the installed run configurations are visible.

## Optional profiles

Ask the dependency planner which extras the selected workflow needs:

```bash
uv run quick-convert requirements train_vq_asr_librispeech
```

See [dependency planning](../config/dependencies.md) for profile selection,
compatibility constraints, and environment checks.

For the VQ-ASR reference workflow:

```bash
uv sync --extra transformers --extra asr --extra training --extra manifests
```

Some profiles intentionally conflict because their upstream packages require incompatible dependency versions. uv enforces those declared conflicts during installation.

## What installation does not include

Optional model checkpoints and datasets are not bundled. A command may still download a model on first use; each workflow guide identifies those cases.

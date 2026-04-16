## Quick Start

This project provides command-line entrypoints for common workflows such as anonymization and ASV training.

After installation, you can run commands with `uv run ...`.

### Check available commands

```bash
uv run anonymize --help
uv run train-asv --help
```

### Run anonymization

Use the anonymize command with a config alias and optional Hydra overrides:

```bash
uv run anonymize <config-alias> [hydra overrides...]
```

Example:


## Installation

This project uses [uv](https://github.com/astral-sh/uv) for Python environment management. Make sure you have it installed before proceeding.

```bash
https://github.com/benluks/quick-convert
cd quick-convert
uv sync
```


## CLI Usage

This project provides simplified CLI entrypoints for running different pipelines (e.g., anonymization, ASV training, evaluation) using Hydra configs under `configs/run/`.

### Basic Pattern

All commands follow the same structure:

```bash
uv run <command> <config-alias> [hydra overrides...]
<command> → the pipeline you want to run (e.g., anonymize, train_asv, eval_asv)
<config-alias> → the suffix of a config file in configs/run/
[hydra overrides...] → optional Hydra overrides (key=value)
```

---

### Examples

#### Anonymization
```bash
uv run anonymize knnvc_clac target_id=6081
```

Uses config:

`configs/run/anonymization_knnvc_clac.yaml`

---

#### Train ASV Model
```bash
uv run train_asv clac asv.overrides.batch_size=32
```

Uses config:

`configs/run/train_asv_clac.yaml`

---

#### Evaluate ASV
```bash
uv run eval_asv clac
```

Uses config:

`configs/run/eval_asv_clac.yaml`

---

### How Config Resolution Works

Each command maps to:

```bash
configs/run/<prefix>_<config-alias>.yaml
```

Where:

* prefix is usually the same as the command
* exception:
    * `anonymize` → `anonymization_*`

So:

```bash
uv run anonymize knnvc_clac
```

→

`configs/run/anonymization_knnvc_clac.yaml`

---

### Hydra Overrides

You can pass any Hydra override directly:

```bash
uv run anonymize knnvc_clac \
  target_id=6081 \
  pipeline.out_dir=foo \
  hydra.job.chdir=false
```
---

### Getting Help

Running a command without arguments will show available config aliases:

```bash
uv run anonymize
```

---

### Notes
* Commands are thin wrappers around modules in `quick_convert/cli/`
* All heavy lifting is still handled by Hydra + your existing pipeline code
* This interface is just a cleaner alternative to:
```bash
uv run python -m quick_convert.cli.<module> \
  --config-name run/<full_config_name> ...
  ```
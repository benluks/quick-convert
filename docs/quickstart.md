# Quick-Convert

A (someday) comprehensive library for running, training, and evaluating speech privacy models.

## Quick Start

This project provides command-line entrypoints for common workflows such as anonymization and ASV training.

After installation (with `uv sync`), you can run commands with `uv run ...`.

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

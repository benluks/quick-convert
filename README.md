# Quick Convert

Quick Convert is a Python library and Hydra-driven toolkit for speech privacy
research: data preparation, feature extraction, training, evaluation, and voice
conversion.

**[Documentation](https://benluks.github.io/quick-convert/)**

## Start here

- [Install Quick Convert](docs/getting-started/installation.md)
- [Understand the architecture](docs/getting-started/concepts.md)
- [Run a supported workflow](docs/guides/workflows.md)
- [Follow the VQ-ASR quickstart](docs/quickstart.md)
- [Use the Python API](docs/library/python-api.md)

A development checkout uses Python 3.11+ and uv:

```bash
uv sync --group dev
uv run quick-convert --help
```

Optional requirements follow the components selected by a run. Inspect them
before installing backends or downloading models:

```bash
uv run quick-convert requirements train_vq_asr_librispeech
```

See [dependency planning](docs/config/dependencies.md) for overrides, profiles,
and compatibility constraints. Datasets and pretrained checkpoints are not
bundled with the package.

## Contribute

Read the [extension guide](docs/extending/index.md) for component and system
contracts, and [AGENTS.md](AGENTS.md) for repository conventions and verification
commands. The [architecture backlog](docs/maintenance/architecture_backlog.md)
records remaining work; the linked guides describe current behavior.

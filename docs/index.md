# Quick Convert

Quick Convert is a composable speech-privacy research library with Python APIs and Hydra-driven workflows for data, feature extraction, training, evaluation, anonymization, and inference.

## Choose a path

### Run a supported workflow

1. [Install Quick Convert](getting-started/installation.md)
2. Learn the [core concepts](getting-started/concepts.md)
3. Follow the [VQ-ASR quickstart](quickstart.md) or choose from the
   [supported workflows](guides/workflows.md)

### Use the Python library

- [Python API guide](library/python-api.md)
- [Data and resources](data/index.md)
- [Configuration reference](config/reference.md)

### Extend the library

- [Design philosophy](design_philosophy.md)
- [Extension guide](extending/index.md)
- [Components](components/index.md)
- [API reference and docstring policy](reference/api-docstrings.md)

## Supported spine

Data discovers audio and attaches named resources. Components implement reusable pieces. Systems expose complete task behavior. Training adds objectives and optimization around systems. Pipelines orchestrate workflows and persistence. Inference artifacts store a resolved system config and its weights.

Executable composition roots live in `quick_convert/configs/run/`. Optional model downloads and full jobs remain environment-dependent; the fast suite verifies composition and library contracts with lightweight substitutes.

Maintenance records document historical migrations and deferred work. They are
useful context, but the guides above define the supported usage surface.

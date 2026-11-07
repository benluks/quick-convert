# Quick Convert

Quick Convert combines Python APIs with Hydra-driven speech privacy workflows.
Start with the overview, choose a task guide, and use references for details.

## Start here

1. [Install](getting-started/installation.md) the base package and required backends.
2. Read [Core concepts](getting-started/concepts.md) for structure and interactions.
3. Run the [VQ-ASR quickstart](quickstart.md), or choose a [supported workflow](guides/workflows.md).

## Task guides

| Goal | Guide |
| --- | --- |
| Prepare data and features | [Data and resources](data/index.md), [precomputation](guides/precomputation.md), [fitting](guides/fitting.md) |
| Train a system | [VQ-ASR quickstart](quickstart.md), [training runs](guides/training.md), [paired reconstruction](guides/paired-reconstruction.md) |
| Load or export a model | [Inference](guides/inference.md) |
| Convert speech | [Voice conversion](guides/voice-conversion.md), [reference conditioning](guides/reference-conditioning.md), [outputs and resuming](guides/conversion-outputs.md) |
| Evaluate speech outputs | [Utterance evaluation](guides/evaluation.md), [speaker verification](guides/asv.md), [frame diagnostics](guides/frame-diagnostics.md) |
| Work with codec representations | [Native codecs](guides/native-codecs.md) |
| Work directly in Python | [Python API](library/python-api.md) |

## Extension and reference

| Question | Reference |
| --- | --- |
| Why are responsibilities separated this way? | [Design philosophy](design_philosophy.md) |
| How do I add or change a piece? | [Extension guide](extending/index.md) |
| What are the model and feature contracts? | [Components](components/index.md) |
| How are configs composed and overridden? | [Configuration](config/reference.md) |
| Which optional packages do I need? | [Dependency planning](config/dependencies.md) |
| What does the pipeline own? | [Lifecycle and batching](pipelines/index.md) |
| Where should API details be documented? | [Docstring policy](reference/api-docstrings.md) |

Run configs under `quick_convert/configs/run/` define supported executable
compositions. Model downloads, full datasets, and optional integrations remain
environment-dependent. The [architecture backlog](maintenance/architecture_backlog.md)
tracks remaining work; it does not override current contracts in the guides.

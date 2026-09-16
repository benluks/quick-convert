# quick-convert documentation

`quick-convert` is a modular framework for building speech privacy experiments from reusable datasets, systems, components, and executable pipelines.

Experiments are configured through Hydra, allowing architectures, datasets, feature extractors, trainers, and evaluation systems to be composed without rewriting the surrounding infrastructure.

## Start here

New users should begin with the [Quickstart](quickstart.md).

The Quickstart starts from a downloaded copy of LibriSpeech and walks through a complete workflow:

1. training a SentencePiece tokenizer;
2. precomputing token IDs;
3. building a CSV manifest;
4. training a VQ-ASR model from that manifest.

It also introduces the main abstractions used throughout the project, including datasets, resources, pipelines, trainers, and Hydra configuration composition.

[**Begin the Quickstart →**](quickstart.md)

## Core concepts

`quick-convert` distinguishes three architectural roles: pipelines orchestrate
workflows, systems implement task-level capabilities, and components provide
focused reusable building blocks. These roles define ownership and dependency
direction; a preprocessing pipeline may use a component directly when no
task-level system is needed.

⚠️ WARNING: Most of the following pages don't exist yet. ⚠️

### [Pipelines](pipelines/index.md)

Pipelines define complete executable workflows, such as:

* training;
* evaluation;
* anonymization;
* feature precomputation;
* manifest generation.

Pipelines coordinate configuration, data loading, execution, and output handling.

### [Systems](systems/index.md)

Systems implement task-level capabilities, such as:

* automatic speech recognition;
* automatic speaker verification;
* speech anonymization and voice conversion.

A system's task contract is independent of the particular components used to
implement it.

### [Components](components/index.md)

Components are reusable model and signal-processing building blocks. They are analogous to PyTorch `nn.Module` objects and may be composed recursively.

Examples include:

* encoders and decoders;
* self-supervised speech models;
* feature extractors;
* speaker embedding models;
* neural network layers;
* losses.

## Documentation

### Using the framework

* [Installation](installation.md)
* [Quickstart](quickstart.md)
* [Configuration and Hydra](configuration.md)
* [Running pipelines](pipelines/index.md)

### Data

* [Datasets](data/datasets.md)
* [Resources](data/resources.md)
* [Manifest datasets](data/manifests.md)
* [Dataloading](data/dataloading.md)

### [Pipelines](pipelines/index.md)

* [Training](pipelines/training.md)
* [Evaluation](pipelines/evaluation.md)
* [Feature precomputation](pipelines/precompute.md)
* [Building manifests](pipelines/build-manifest.md)
* [Anonymization](pipelines/anonymization.md)

### [Systems](systems/index.md)

* [Automatic speech recognition](systems/asr.md)
* [Automatic speaker verification](systems/asv.md)
* [Anonymization and voice conversion](systems/anonymization.md)

### [Components](components/index.md)

* [Encoders and decoders](components/encoders-decoders.md)
* [Self-supervised models](components/ssl.md)
* [Feature extractors](components/feature-extractors.md)
* [Quantizers](components/quantizers.md)
* [Losses](components/losses.md)

### Development: TODO

* [Repository structure](development/repository-structure.md)
* [Adding a dataset](development/adding-a-dataset.md)
* [Adding a component](development/adding-a-component.md)
* [Adding a pipeline](development/adding-a-pipeline.md)
* [Contributing](development/contributing.md)

## How experiments are assembled

Most experiments begin with a run configuration under `configs/run/`.

A run configuration selects and combines the relevant configuration groups:

```yaml
defaults:
  - /global: default
  - /pipeline: training
  - /trainer: vq_asr
  - /dataset@train_dataset: manifest
  - _self_
```

Hydra composes these files into a complete runtime configuration. The selected pipeline is then instantiated and executed.

```text
Run configuration
       │
       ▼
Hydra composition
       │
       ▼
    Pipeline
       │
       ▼
  Dependencies
       │
       ▼
     Outputs
```

## Project status

`quick-convert` is an active research codebase.

Some framework-specific training adapters still live beneath
`quick_convert/pipelines/training`; these are expected to move into a dedicated
training package. Compatibility re-exports also remain for anonymization
systems that previously lived beneath `pipelines`.

Optional dependency groups are also being revised and tested. Consult `pyproject.toml` when installing dependencies for a specific workflow.

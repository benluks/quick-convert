# quick-convert

`quick-convert` is a modular framework for speech privacy research. It provides reusable components for datasets, feature extraction, preprocessing, training, and evaluation, allowing new experiments to be assembled through Hydra configuration rather than extensive code changes.

📚 **Documentation:** https://benluks.github.io/quick-convert/

Choose the shortest path for your goal:

- [Install and learn the core concepts](https://benluks.github.io/quick-convert/getting-started/installation.html)
- [Run a supported workflow](https://benluks.github.io/quick-convert/guides/workflows.html)
- [Use the Python library directly](https://benluks.github.io/quick-convert/library/python-api.html)
- [Extend the library](https://benluks.github.io/quick-convert/extending/)

The framework is designed around composition. Datasets, resources, models, feature extractors, trainers, and pipelines are all interchangeable, making it straightforward to build new workflows while reusing existing infrastructure.

## Features

* **Hydra-based configuration** for reproducible, composable experiments.
* **Flexible datasets** with pluggable resource providers.
* **Preprocessing pipelines** for manifest generation, feature precomputation, and tokenizer training.
* **Training pipelines** for speech models and auxiliary components.
* **Evaluation pipelines** for benchmarking and analysis.
* **Reusable components**, including encoders, decoders, feature extractors, SSL models, quantizers, and losses.

Many components rely on optional dependencies. These are grouped into extras so that only the libraries required for a particular workflow need to be installed.   

| Extra                  | Description                          |
| ---------------------- | ------------------------------------ |
| `manifests`            | Data-frame utilities for splitting manifests |
| `transformers`         | Hugging Face-backed models, including W2V-BERT and WavLM |
| `w2vbert`              | Compatibility alias for `transformers` |
| `whisper`              | Whisper ASR model                    |
| `sentencepiece`        | SentencePiece tokenization           |
| `wer`                  | JIWER word-error-rate evaluation     |
| `asr`                  | Compatibility bundle containing `sentencepiece` and `wer` |
| `training`             | Lightning training with Matplotlib, W&B, and TensorBoard logging |
| `lightning`            | Compatibility alias for `training` |
| `cosyvoice`            | CosyVoice reconstruction decoder dependencies |
| `emotion2vec`          | FunASR-backed emotion2vec feature extraction |
| `conditional-rvq`     | Compatibility alias for `emotion2vec` |
| `mpm`                  | Masked Prosody Model feature extraction |
| `espnet-wavlm-joint`   | ESPnet WavLM speaker encoder |
| `pyannote`             | Experimental pyannote WeSpeaker integration |
| `dac`                  | Experimental DAC content encoder; unrelated to the DAC-style RVQ layer |
| `web`                  | Legacy Flask interface; currently unverified |

Normally, when you import a module, you'll get a `ModuleNotFoundError` if the requisite dependencies are missing. Check out `pyproject.toml` to see which extras are needed to run whatever it is you're trying to run.

For example:

```bash
uv sync --extra transformers --extra asr --extra training
```

The current reference workflows require these extras:

| Workflow | Extras |
| -------- | ------ |
| Build a LibriSpeech manifest | none |
| Precompute W2V-BERT content | `transformers` |
| Train a SentencePiece tokenizer | `sentencepiece`, `training` |
| Evaluate Whisper ASR | `whisper`, `wer` |
| Train VQ-ASR with W2V-BERT | `transformers`, `asr`, `training` |
| Train SSL reconstruction with CosyVoice | `transformers`, `cosyvoice`, `training` |

> **Note**
>
> The declared conflicts are verified against the current dependency metadata.
> `dac` conflicts with `espnet-wavlm-joint` over incompatible Protobuf ranges;
> `mpm` conflicts with `espnet-wavlm-joint` over NumPy 1.x versus 2.x.
> See the [dependency profile maintenance notes](docs/maintenance/dependency_extras.md)
> for support status and open decisions.

## Quickstart

The recommended end-to-end introduction is the [Quickstart guide](https://benluks.github.io/quick-convert/quickstart.html). For a shorter task lookup, use the [workflow guide](https://benluks.github.io/quick-convert/guides/workflows.html).

Starting from a downloaded LibriSpeech dataset, it walks through:

1. training a SentencePiece tokenizer;
2. precomputing token IDs;
3. building a manifest dataset;
4. training a VQ-ASR model.

Along the way, it introduces the core abstractions used throughout the project:

* datasets;
* resources;
* pipelines;
* trainers;
* Hydra configuration composition.

➡ [**See the Quickstart guide.**](https://benluks.github.io/quick-convert/quickstart.html)

## Design philosophy

quick-convert separates workflow orchestration from task behavior and reusable
implementation pieces:

- **Pipelines** execute workflows over datasets and persist outputs.
- **Systems** provide complete task-level capabilities through a library API.
- **Components** are focused building blocks used to construct systems or
  specialized workflows.

These are roles and dependency boundaries, not a requirement that every
workflow instantiate all three. For example, feature precomputation may apply a
feature-extractor component directly. Training inserts a trainer and a
framework-specific training module around a system without making either one a
task system.

### Pipelines

Pipelines define complete executable workflows under `quick_convert/pipelines`.

Examples include:

- model training, independent of the particular task system;
- evaluation;
- dataset anonymization; and
- feature precomputation.

A pipeline coordinates data loading, systems, output handling, and runtime configuration.

### Systems

Systems implement a complete task-level capability under
`quick_convert/systems`. Their public inference behavior does not depend on a
pipeline or training framework.

Examples include:

- automatic speech recognition;
- speech reconstruction; and
- anonymization or voice conversion.

A system may combine several models or call another system when that dependency
is itself task-level.

### Components

Components are reusable, focused building blocks. Many are recursive PyTorch
modules, but the role also includes stateless signal processing and feature
extraction utilities.

Examples include:

encoders and decoders;
self-supervised speech models;
neural network layers;
losses;
feature extractors;
x-vector extractors

This separation allows low-level components to be reused across different systems, while pipelines remain focused on how those systems are trained, evaluated, or applied.

Most experiments select a pipeline and its dependencies through Hydra. Task
workflows configure a system directly at `system`; training workflows pass that
same system to a framework-specific training module.

## Python library use

The package can be used without a pipeline. For example, load an exported
system as a plain inference object:

```python
from quick_convert.inference import load_inference_artifact

system = load_inference_artifact("models/vq-asr", map_location="cpu")
```

Installed Hydra recipes live under `quick_convert/configs/`; `quick-convert
--help` lists their composition roots. Pipelines are orchestration conveniences,
not a prerequisite for importing datasets, systems, or components.

## Contributing

Contributions are welcome. Bug reports, feature requests, documentation improvements, and pull requests are all appreciated.

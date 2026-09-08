# quick-convert

`quick-convert` is a modular framework for speech privacy research. It provides reusable components for datasets, feature extraction, preprocessing, training, and evaluation, allowing new experiments to be assembled through Hydra configuration rather than extensive code changes.

📚 **Documentation:** https://benluks.github.io/quick-convert/

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
| `w2vbert`              | W2V-BERT feature extraction          |
| `whisper`              | Whisper ASR model                    |
| `asr`                  | SentencePiece tokenization and JIWER evaluation |
| `lightning`            | Lightning training and associated logging tools |
| `cosyvoice`            | CosyVoice reconstruction decoder dependencies |
| `espnet-wavlm-joint`   | ESPnet WavLM implementation             |
| `pyannote`             | For the pyannote WeSpeaker implementation      |
| `dac`                  | Descript Audio Codec support         |
| `web`                  | Web interface components. I think she's currently broken.             |

* **Hydra-based configuration** for reproducible, composable experiments.
* **Flexible datasets** with pluggable resource providers.
* **Preprocessing pipelines** for manifest generation, feature precomputation, and tokenizer training.
* **Training pipelines** for speech models and auxiliary components.
* **Evaluation pipelines** for benchmarking and analysis.
* **Reusable components**, including encoders, decoders, feature extractors, SSL models, quantizers, and losses.

Normally, when you import a module, you'll get a `ModuleNotFoundError` if the requisite dependencies are missing. Check out `pyproject.toml` to see which extras are needed to run whatever it is you're trying to run.

For example:

```bash
uv sync --extra w2vbert --extra asr --extra lightning
```

The current reference workflows require these extras:

| Workflow | Extras |
| -------- | ------ |
| Build a LibriSpeech manifest | none |
| Precompute W2V-BERT content | `w2vbert` |
| Train VQ-ASR with W2V-BERT | `w2vbert`, `asr`, `lightning` |
| Train SSL reconstruction with CosyVoice | `w2vbert`, `cosyvoice`, `lightning` |

> **Note**
>
> Some extras depend on conflicting versions of third-party libraries and therefore cannot be installed together. See `pyproject.toml` for the defined compatibility groups. I can't promise it's up-to-date. I didn't fully understand how conflicts worked back when I started writing it. Currently in the process of fixing it, and writing tests.

## Quickstart

The recommended introduction to the framework is the [Quickstart guide](https://benluks.github.io/quick-convert/quickstart.html).

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

quick-convert is organized into three conceptual layers:

Pipelines
    │
    ▼
Systems
    │
    ▼
Components

### Pipelines

Pipelines define complete executable workflows. These can be found under `quick_convert/pipelines/{[PIPELINE_NAME]/pipeline.py,[PIPLEINE_NAME].py}`.

Examples include:

Model training pipelines, which is agnostic to the task (system) and architecture (components);
Evaluation which is similarly agnostic;
Anonymization a dataset;
Precomputing features (although maybe this should be under an "inference" pipeline, we'll see);

A pipeline coordinates data loading, systems, output handling, and runtime configuration.

### Systems

Systems implement a complete task-level capability. They're found under `quick_convert/pipelines/{[PIPELINE_NAME]/[SYSTEM_NAME]/...}`. I put ASR in a dedicated systems folder `quick_convert/systems/asr`. That's the plan for the future. I just haven't done the refactoring yet.

Examples include:

ASR system, invariant to the exact architecture;
Automatic Speaker Verification (ASV);
Anonymization/Voice Converstion.

A system may combine multiple models, and, frankly, a model may utilize multiple systems.

### Components

Components are the reusable building blocks from which systems are constructed. This are analogous to pytorch `nn.Module`s, and are similarly recursive.

Examples include:

encoders and decoders;
self-supervised speech models;
neural network layers;
losses;
feature extractors;
x-vector extractors

This separation allows low-level components to be reused across different systems, while pipelines remain focused on how those systems are trained, evaluated, or applied.

Configuration
      │
      ▼
   Pipeline
      │
      ▼
    System
      │
      ▼
  Components
      │
      ▼
    Output

Most experiments in quick-convert are created by selecting a pipeline, configuring a system, and composing its components through Hydra.

## Contributing

Contributions are welcome. Bug reports, feature requests, documentation improvements, and pull requests are all appreciated.

# Quick Convert

Quick Convert is a composable speech-privacy research library with Python APIs and Hydra-driven workflows for data, feature extraction, training, evaluation, anonymization, and inference.

## Start here

- [Installation](installation.md)
- [VQ-ASR quickstart](quickstart.md)
- [Design philosophy](design_philosophy.md)
- [Data and resources](data/index.md)
- [Components](components/index.md)
- [Hydra configuration](config/hydra_structure.md)

## Supported spine

Data discovers audio and attaches named resources. Components implement reusable pieces. Systems expose complete task behavior. Training adds objectives and optimization around systems. Pipelines orchestrate workflows and persistence. Inference artifacts store a resolved system config and its weights.

Executable composition roots live in `quick_convert/configs/run/`. Optional model downloads and full jobs remain environment-dependent; the fast suite verifies composition and library contracts with lightweight substitutes.

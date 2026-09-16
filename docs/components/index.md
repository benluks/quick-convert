# Components

Components are reusable building blocks beneath task-level systems:

- encoders and decoders;
- feature extractors and SSL backends;
- quantizers and neural-network layers;
- losses;
- speaker encoders and generators;
- low-level acoustic features.

Components should expose explicit tensor and length contracts. They may compose recursively, but should not own datasets, output directories, training loops, or workflow policy.

Complete inference behavior belongs in [systems](../design_philosophy.md#systems); orchestration belongs in [pipelines](../design_philosophy.md#pipelines). Vendored implementations under `quick_convert.external` are private dependencies reached through first-party adapters.

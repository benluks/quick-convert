# Components

Components are reusable building blocks beneath task-level systems:

- encoders and decoders;
- feature extractors and SSL backends;
- quantizers and neural-network layers;
- losses;
- speaker encoders and generators;
- low-level acoustic features.

Components should expose explicit tensor and length contracts. They may compose recursively, but should not own datasets, output directories, training loops, or workflow policy.

## Content encoders

Every `ContentEncoder` declares its required waveform `sample_rate`, its
`frame_hz` when the representation has a regular frame timebase, and a batch
entry point `forward(AudioBatch) -> ContentFeatures`. Returned
`ContentFeatures` repeat the effective `frame_hz` alongside exact valid frame
lengths so downstream tools can align representations by time rather than by
proportional sequence position. Utterance-level representations use
`frame_hz=None`.

`frame_hz` describes the spacing between successive frames. It does not imply
that a frame is centered at time zero or describe the frontend's receptive
field; consumers that require sample-exact boundaries should keep those
distinctions explicit.

Complete inference behavior belongs in [systems](../design_philosophy.md#systems); orchestration belongs in [pipelines](../design_philosophy.md#pipelines). Vendored implementations under `quick_convert.external` are private dependencies reached through first-party adapters.

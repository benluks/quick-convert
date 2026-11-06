# Pipelines and inference

Pipelines are executable workflows. They assemble datasets and the smallest
appropriate executable dependency—usually a task system, or directly a focused
component for preprocessing—iterate over data, and persist task-specific
outputs. They should not contain model architecture or backend-specific
inference logic.

The pipeline–system–component model describes roles rather than mandatory
nesting. Training additionally uses a trainer backend and a framework adapter:
the pipeline owns the run, the trainer owns framework execution, the adapter
owns objectives and hooks, and the wrapped system remains usable for inference.

The current inference-oriented pipelines share data contracts, but they do not
yet share enough lifecycle behavior to justify a common base class.

| Pipeline | Inference entry point | Output contract | Persistence |
| --- | --- | --- | --- |
| Feature precomputation | `extractor.extract_batch(AudioBatch)` | one feature object per input sample | feature files, manifest, and cache identity |
| Utterance evaluation | `system.get_labels(AudioBatch)` | one prediction per input sample | per-utterance CSV, aggregate JSON, and provenance |
| Speaker verification | `speaker_encoder.encode_batch(AudioBatch)` | embeddings scored over enrollment–test trials | aggregate JSON and provenance |
| Speech conversion | `system.convert_batch(AudioBatch, target=prepared)` | `GeneratedAudio` with one waveform per input | audio, provenance, and manifests |

Conversion systems live under `quick_convert.systems.conversion`. A system
prepares explicit voice conditioning; `SpeechConversionPipeline` coordinates
datasets, target selection, output files, and resuming. Anonymization is a target
selection and evaluation policy over conversion, rather than a separate model hierarchy.

## Execution lifecycle

The shared CLI runner composes a run config and instantiates `cfg.pipeline`.
It calls `prepare()`, `write_config()`, and `log_config()` in that order when
the pipeline exposes them, then invokes `pipeline.run(**cfg.run)`. Preparation allows the backend to determine its final
run directory before the resolved configuration is persisted. Direct calls to
`TrainingPipeline.run()` prepare automatically. See [configuration](../config/reference.md)
for aliases and composition slots.

## Batch contracts

`BaseDataset.make_dataloader()` collates samples into `AudioBatch`. When audio
is loaded, `AudioBatch.waveforms` is padded and `AudioBatch.lengths` records the
valid number of samples in each waveform. Inference code must use these lengths
instead of treating the padded time dimension as valid audio.

Batch-producing systems and extractors must return one logical output per input
sample. Pipelines validate this cardinality before associating outputs with
sample metadata or writing files.

Generated waveform batches use `GeneratedAudio`, which contains padded
waveforms, their valid lengths, and their common sample rate. This is the output
contract required for correct batched waveform persistence.

## Conversion batching

Systems expose `convert_batch(AudioBatch, target=prepared) -> GeneratedAudio`.
KNN-VC and ASR-BN share an exact sequential adapter over their single-item
`convert()` methods. It records lengths before padding and uses unpadded source
samples. A batch size greater than one enables data orchestration without implying
vectorized backend inference. SSL conversion retains its native batch path.

The pipeline persists only each waveform's reported valid length.

## Appropriate shared infrastructure

The pipelines can share small, behaviorally exact utilities when repetition is
established, such as validating that an output collection has one item per
sample. Progress descriptions, reference-dataset coordination, output paths,
resume behavior, and file formats remain pipeline-specific concerns.

A general `BasePipeline` is intentionally deferred. Training, evaluation,
precomputation, manifest construction, and anonymization have different
lifecycles; inheriting from one base class would currently provide naming
uniformity without a stable behavioral contract.

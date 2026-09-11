# Pipelines and inference

Pipelines are executable workflows. They assemble datasets and systems or
components, iterate over data, and persist task-specific outputs. They should
not contain model architecture or backend-specific inference logic.

The current inference-oriented pipelines share data contracts, but they do not
yet share enough lifecycle behavior to justify a common base class.

| Pipeline | Inference entry point | Output contract | Persistence |
| --- | --- | --- | --- |
| Feature precomputation | `extractor.extract_batch(AudioBatch)` | one feature object per input sample | tensor files and `manifest.jsonl` |
| Evaluation | `system.get_labels(AudioBatch)` | one prediction per input sample | per-utterance CSV and aggregate JSON |
| Anonymization | `anonymizer.anonymize(...)` | one waveform for one input sample | waveform files |

Anonymizers expose a lightweight library API: `anonymize()` accepts either a
file path or a waveform tensor and returns a waveform tensor. A tensor may
provide its `sample_rate`; when omitted, it is assumed to already use the
anonymizer's declared input rate. Files carry their own sample-rate metadata.
Users do not need to construct dataset samples or batches for single-item use.
The implementations live under `quick_convert.systems.anonymization`; the
pipeline only coordinates datasets, targets, and persisted outputs.

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

## Why anonymization remains unbatched

The current anonymizers expose single-item, backend-specific methods and do not
return `GeneratedAudio`. `AnonymizationPipeline` therefore rejects
`batch_size > 1` explicitly. Silently batching these implementations would make
padding indistinguishable from valid generated audio.

The path to batching is:

1. give anonymization systems a batch-oriented entry point accepting
   `AudioBatch`;
2. require that entry point to return `GeneratedAudio`;
3. write each waveform only up to its reported valid length;
4. then make `AnonymizationPipeline` iterate over a dataloader.

This should be implemented for a current supported anonymization system before
extracting a generic inference protocol. The protocol should follow proven
call sites rather than forcing feature extraction, label prediction, and audio
generation into an artificial common return type.

## Appropriate shared infrastructure

The pipelines can share small, behaviorally exact utilities when repetition is
established, such as validating that an output collection has one item per
sample. Progress descriptions, reference-dataset coordination, output paths,
resume behavior, and file formats remain pipeline-specific concerns.

A general `BasePipeline` is intentionally deferred. Training, evaluation,
precomputation, manifest construction, and anonymization have different
lifecycles; inheriting from one base class would currently provide naming
uniformity without a stable behavioral contract.

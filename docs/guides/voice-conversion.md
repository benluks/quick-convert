# Reference speech conversion

Load an inference system and prepare a target once, then convert source batches:

```python
from quick_convert.inference import load_inference_system
from quick_convert.systems.anonymization.targets import ReferenceSpeech

system = load_inference_system("/models/export")
system.eval()
target = system.prepare_target(ReferenceSpeech(["/audio/reference.wav"], prompt_seconds=3.0))
generated = system.convert_batch(source_batch, target=target)
```

SSL reconstruction systems support reference content/mels and named conditioning
streams. Models with a global speaker encoder aggregate reference embeddings;
`SpeakerEmbeddingAggregation("duration_weighted")` weights by recording duration.
`SpeechPromptBuilder` can assemble a speech-only prompt across recordings. Missing
named streams raise an error unless `allow_missing_streams=True` is explicit.
Prepared targets belong to the system instance that created them.

KNN-VC supports the same preparation/conversion calls with
`ReferenceSpeech(["/audio/reference.wav"])`. It uses a matching set and rejects
prompt and aggregation options. Its existing anonymization interface remains
available.

For a dataset, use `SpeechConversionPipeline(system, dataset, target, out_dir)`.
`DatasetReferenceTargets` selects fixed or seeded random speaker groups from a
named speaker resource; random selection can exclude the source speaker. Grouped
conversion caches prepared targets with a bounded cache. `max_samples` and
`sample_seed` select a reproducible shuffled prefix without changing global RNG
state. Optional `inference_cfg_rate` is forwarded to systems that support it.

The executable roots are `convert_reference_manifest`,
`convert_reference_librispeech`, and `convert_reference_librispeech_random`:

```bash
quick-convert convert_reference_manifest source=/models/export \
  manifest_path=/data/sources.yaml 'target_paths=[/audio/reference.wav]'
quick-convert convert_reference_librispeech source=/models/export target_id=6081
```

These presets assemble a three-second speech prompt. Choose a compatible SSL
system and configure dataset roots for your environment. For KNN-VC, use the Python
API above or explicitly clear both prompt options. See
[output records](conversion-outputs.md) for persistence and resuming.

# Reference speech conversion

Load an inference system and prepare a target once, then convert source batches:

```python
from quick_convert.inference import load_inference_system
from quick_convert.systems.conversion.targets import ReferenceSpeech

system = load_inference_system("/models/export")
system.eval()
target = system.prepare_target(ReferenceSpeech(["/audio/reference.wav"], prompt_seconds=3.0))
generated = system.convert_batch(source_batch, target=target)
```

SSL reference conversion always encodes the reference prompt's content.
Named streams such as CAMPPlus or mel provide additional conditioning on the
reference content timeline. The decoder discards the generated reference span
and returns only the source continuation. Models with a global speaker encoder aggregate reference embeddings;
`SpeakerEmbeddingAggregation("duration_weighted")` weights by recording duration.
`SpeechPromptBuilder` can assemble a speech-only prompt across recordings. Missing
named streams raise an error unless `allow_missing_streams=True` is explicit.
Prepared targets belong to the system instance that created them.

KNN-VC supports the same preparation/conversion calls with
`ReferenceSpeech(["/audio/reference.wav"])`. It uses a matching set and rejects
prompt and aggregation options. Use explicit prepared targets rather than mutable `set_target()` state.

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
  manifest_path=/data/sources.csv 'target_paths=[/audio/reference.wav]'
quick-convert convert_reference_librispeech source=/models/export target_speaker=6081
```

These presets assemble a three-second speech prompt. Choose a compatible SSL
system and configure dataset roots for your environment. For KNN-VC, use the Python
API above or explicitly clear both prompt options. See
[output records](conversion-outputs.md) for persistence and resuming.

## Conversion systems and migration

| System | Target request |
| --- | --- |
| `SSLReconstructionConversionSystem` | `ReferenceSpeech` or `SourceVoice` |
| `KNNVCConversionSystem` | `ReferenceSpeech` or `SourceVoice` |
| `ASRBNConversionSystem` | `SpeakerIdentity` |

Import systems and requests from `quick_convert.systems.conversion`. For example,
`system.prepare_target(SpeakerIdentity("1069"))` selects an ASR-BN voice;
`system.prepare_target(SourceVoice())` requests source resynthesis on supporting
systems. Each conversion call receives its prepared target explicitly. KNN-VC
and ASR-BN also accept a path or waveform through `convert(audio, target=prepared,
sample_rate=16000)`. Prepared targets are tied to the system instance.

Conversion runs and the verb command use `convert`, including `convert_knnvc_clac`
and `convert_knnvc_librispeech`. KNN-VC prepares the target matching set once;
no separate feature preparation is required.

## KNN-VC with LibriSpeech

From a checkout, install with `uv sync`, then run:

```bash
uv run quick-convert convert_knnvc_librispeech \
  root=/data/LibriSpeech source_split=test-clean source_speaker=1089 \
  target_speaker=1069
```

Omit `source_speaker` to convert the whole source split. `root` is the directory
containing the split folders. `dataset.pattern` remains directly overridable;
the preset defaults it to the selected speaker's chapter recordings. References
are selected from `target_dataset` using its speaker resource; `target_root`
defaults to `root`. References are discovered across all available splits below
that root by default, independently of `source_split`. To restrict references,
pass e.g. `'target_splits=[train-clean-100,dev-clean]'`. Only explicitly selected
splits must exist. Selection reads speaker metadata without decoding reference
audio; reference scope and selected paths are recorded in provenance.
For CLAC sources and LibriSpeech references, use `convert_knnvc_clac` with
`reference_dataset=librispeech`, `dataset.root`, `target_root`, and `target_speaker`.
The shared KNN-VC recipe has no default reference corpus or audio format.
A custom `target_dataset` can use any supported audio format and resource
providers; `target_speaker_resource` names the speaker resource (default `spkid`).
KNN-VC defaults to `target_max_utterances=10`, `target_max_seconds=60`,
`target_shuffle=true`, and `target_seed=0`. Candidates are sorted and then
shuffled reproducibly; selection stops at either limit before feature extraction.
Duration is read from audio headers without decoding waveforms. Whole utterances
are kept, so the final utterance may take the total beyond the time budget.
Shorter available sets are accepted and selected counts/duration are logged.
Set either limit to `null` to disable it, or both to select every matching recording.
Selected paths and selection settings are recorded in conversion provenance.

The pretrained backend downloads code and weights through Torch Hub on first
use. Its current inference imports are covered by the base installation; no
additional optional profile is needed for this path. GPU execution is the
default; select `device=cpu` for a small CPU run. These instructions describe
the configuration interface; full pretrained execution is environment-dependent.

The former `systems/anonymization`, `systems/reconstruction`, and
`pipelines/anonymization` modules are removed. Update direct Python imports and
custom Hydra configs: use `system/conversion@system`, top-level `system` and
`target`, and `SpeechConversionPipeline`. Saved run, checkpoint, and inference
artifact compositions migrate known old `_target_` paths during loading;
parameter keys are unchanged. This does not restore arbitrary pickled Python
objects or automatically rewrite custom workflow YAML. Old output journals
remain readable; changed target descriptions can require a fresh output directory.

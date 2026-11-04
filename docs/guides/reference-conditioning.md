# Reference prompts and conditioning streams

Reference conditioning supplies speaker or acoustic context while the source
content remains the linguistic input. The first-party CosyVoice adapter supports
an aligned content/mel reference and independent named prefix streams.

## Run prefix-conditioned reconstruction

Start with mel-only conditioning on the existing W2V-BERT DiT system:

```bash
quick-convert requirements train_sslr_w2vbert_dit_streams
quick-convert train_sslr_w2vbert_dit_streams data_root=/data
```

The run samples a target-mel prefix during training and scores only its
continuation. For frame-wise speaker context without a global speaker vector:

```bash
quick-convert train_sslr_w2vbert_dit_campp_prefix data_root=/data
quick-convert train_sslr_w2vbert_cmdiff_campp_prefix data_root=/data
```

The CAMPPlus variants require `campplus-frames`, in addition to the content,
decoder and training profiles selected by `requirements`. Their online speaker
encoder is replaced by a frozen 512-channel CAMPPlus frame encoder. The
`train_sslr_dac_dit_campp_prefix` and `train_sslr_dac_cmdiff_campp_prefix` variants
use frozen online DAC content instead of W2V-BERT. They require `dac` rather
than `transformers`. Model downloads and training data are still required.

`train_sslr_w2vbert_dit_streams_no_xvector` and
`train_sslr_w2vbert_cmdiff_streams_no_xvector` use mel prefixes without a global
speaker vector. Existing reconstruction runs keep their original conditioning.
Each new preset has its own experiment name derived from the executable run
name, so prefix variants do not share the base run's output directory.

## Named streams

`StreamConditioning` declares the ordered stream names and dimensions. For
example:

```python
from quick_convert.components.decoders import StreamConditioning

conditioning = StreamConditioning({"mel": 80, "pitch": 1})
assert conditioning.output_dim == 83
```

Each stream adds an availability channel. Stream names, order and dimensions
are saved in the checkpoint; a different layout is rejected on restoration.
The estimator's `cond_dim` must match the packed width. The public run configs
show the corresponding DiT and CFM compositions; vendored estimator classes
remain implementation details reached through the adapter.

`ConditioningStream(values, lengths)` holds `[batch, time, dim]` values and
int64 valid lengths on the **mel clock**. It performs no automatic resampling.
Streams share the content device and dtype; valid values must be finite.
Missing streams are represented by zero availability, not fabricated context.
Different streams can have different prefix lengths.

During training, ordinary streams cover the complete source duration. The
decoder samples an independent prefix for each, masks independent spans within
those prefixes, and can drop streams entirely. The loss covers the valid frames
after the longest prefix, so an observed prefix cannot directly reduce the
continuation loss. Source linguistic content stays visible when auxiliary
context is dropped. The reserved `mel` stream comes from the decoder's own
target-mel extraction; callers cannot replace it in a training call.

For contextual encoders, crop the waveform **before** feature extraction.
CAMPPlus uses this path: the system samples prefix lengths before calling
`encode_prefixes`. Encoding the full recording and then truncating features
could leak future context. An already cropped training prefix uses
`ConditioningStream(..., is_prefix=True)` and is not sampled a second time.

At inference, pass only the prepared prefixes:

```python
from quick_convert.components.decoders import ConditioningStream

# prefix_mels: [batch, prefix_time, 80]; prefix_lengths: int64 [batch].
result = system.generate_features(
    content_values,
    content_lengths,
    speaker_embedding,
    streams={"mel": ConditioningStream(prefix_mels, prefix_lengths)},
)
```

For a model composed without global speaker conditioning, omit
`speaker_embedding`. Passing a speaker vector to such a decoder raises an error.
Prefix streams are aligned at their ends, with shorter streams left-padded.
Generation returns source-only mels and valid lengths, excluding the prefix.
Validation also uses prefix-only inference rather than complete target streams.

`decoder(..., inference_cfg_rate=0.3)` overrides guidance for that call without
changing the saved default. `None` preserves the configured value; negative or
nonfinite rates are rejected. Named-stream sampling keeps source content in the
null auxiliary branch. `stream_inference_steps` controls the stream sampler's
positive integer step count.

## Aligned content and mel references

For a continuous-feature flow, prepare an explicit reference through the decoder:

```python
reference = decoder.prepare_reference(
    reference_content,
    reference_content_lengths,
    reference_waveforms,
    reference_waveform_lengths,
    sample_rate=16_000,
)
result = decoder(
    source_content,
    source_content_lengths,
    speaker_embedding,
    reference=reference,
)
```

`CosyVoiceReference` stores content features and mels with independent valid
lengths. The adapter validates dimensions, finite values and the token/mel
ratio. `reference.expand(batch_size, device=..., dtype=...)` copies a single
reference without sharing writable storage between rows or with the original.
Batched generation requires equal reference lengths.

Encoder-flow training can pass `reference` to `compute_loss`; reference and
source are compacted into one sequence, with loss only on the source. Native
DiT training retains its own mel-prefix objective; separate-reference training
requires an encoder flow. Named streams are the other supported prefix training
path. An aligned `reference` and named streams cannot be combined in one call.

## Reproducible speech prompt assembly

`SpeechPromptBuilder` consumes recordings in caller-supplied order, removes
quiet intervals, and fills an exact duration using crossfades and edge fades:

```python
from pathlib import Path
from quick_convert.components.reference_prompt import SpeechPromptBuilder

# Each waveform is finite, mono floating-point audio at one common sample rate.
recordings = [(Path("reference.wav"), reference_waveform, 16_000)]
prompt = SpeechPromptBuilder()(recordings, seconds=3.0)
assert prompt.waveform.numel() == 48_000
```

The duration is measured after overlap. Insufficient active speech raises an
error rather than repeating or shortening the prompt. Source intervals and
overlap lengths are retained in `prompt.segments`. Caller waveforms and random
state are preserved. The default `EnergyActivityDetector` uses RMS with an
absolute dBFS threshold; it removes quiet regions rather than identifying all
speech or suppressing loud background noise. A custom detector can supply
ordered, non-overlapping sample intervals through the same callable interface.

## Same-speaker reference resources

Attach a deterministic different recording as a named dataset resource:

```python
from quick_convert.data import ManifestDataset
from quick_convert.data.resources import SameSpeakerReferenceProvider

provider = SameSpeakerReferenceProvider("manifest.csv", speaker_column="spkid", seed=17)
dataset = ManifestDataset("manifest.csv", resource_providers=[provider])
```

The manifest supplies `utt_id`, `path`, `split` and the speaker column.
Candidates stay within the same split and speaker and exclude the source path.
Selection uses a stable hash and sorted candidates, without changing random
state. A group with no different recording is rejected. The resource is named
`reference_path`; the consumer owns loading, cropping and encoding that audio.
The provider requires the `manifests` profile. It adds no fields to `AudioSample`
and does not change dataset loading policy.

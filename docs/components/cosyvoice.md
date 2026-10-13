# CosyVoice reconstruction features

`CosyVoiceSpectrogramGenerator` uses a resolved
`quick_convert.components.feature_extractors.mel.MelSpectrogramConfig` as its
waveform-to-mel contract. Content encoders can continue to consume 16 kHz audio;
the decoder resamples reconstruction targets to the mel specification's rate.
Training targets and validation reference images use the same extraction path.
Each valid waveform is extracted before padding mel frames, so another sample's
length does not change its endpoint features.

For the default pretrained HiFT vocoder, the first-party adapter reads
`feat_extractor`, `hift`, and `f0_predictor` from the companion `cosyvoice3.yaml`.
It reads YAML nodes as data, resolves simple scalar references, and does not
execute Python construction tags. The preprocessing format is Matcha's
reflect-padded, uncentered STFT, log magnitude, and Slaney mel scale and
normalization. Other feature conventions require their own adapter.

The companion configuration supplies the analysis FFT/window, frequency limits,
and mel count. The adapter checks the mel rate, hop, and channels against the
vocoder architecture. Weight shapes alone cannot determine these preprocessing
parameters. Missing fields and unsupported reference expressions fail explicitly.

## Pretrained configuration

The supported reconstruction run uses this contract automatically:

```bash
quick-convert train_sslr_w2vbert_cmdiff data_root=/data
```

For another compatible checkpoint, override the existing decoder fields:

```yaml
system:
  decoder:
    vocoder_repo_id: organisation/model
    vocoder_filename: hift.pt
    vocoder_config_filename: model.yaml
    vocoder_revision: exact-model-commit
```

Only companion metadata is needed to train spectrogram reconstruction. Vocoder
weights remain lazy until waveform generation. Hugging Face snapshot metadata
pins subsequent weight downloads to the same commit. If a repository uses a
different companion filename or schema, supply that filename or a first-party
adapter for that schema; no preprocessing settings are inferred from weights.

## Custom composition

For a locally constructed vocoder, supply its actual input specification to the
vocoder and/or spectrogram generator. For example, a model trained with the
following preprocessing can use:

```yaml
mel_config:
  _target_: quick_convert.components.feature_extractors.mel.MelSpectrogramConfig
  sample_rate: 24000
  n_fft: 1920
  hop_size: 480
  win_size: 1920
  num_mels: 80
  fmin: 0
  fmax: null
```

These values describe that model, rather than defaults for every implementation.
An explicit specification must agree with a supplied or subsequently loaded
vocoder. The flow's configured output frame rate must agree with the mel rate.
The current reconstruction training wrapper aligns one content frame to one mel
frame; configurations requiring temporal upsampling need a corresponding training
alignment implementation.

## Checkpoint compatibility

The resolved specification, pretrained architecture metadata, and model source
are saved as tensor-backed decoder state, so both Lightning checkpoints and
inference artifacts preserve them. Loading saved metadata does not require a
companion download. Explicit conflicting mel specifications are rejected.

Checkpoints predating this metadata are rejected rather than silently resuming
with different targets. They learned the earlier 16 kHz / 8 kHz-upper-limit mel
representation. Migration must explicitly preserve that representation and use
an appropriate vocoder, or deliberately start training with the new feature
contract; simply assigning the new metadata does not repair old weights.

# CosyVoice reconstruction features

For named prefix streams, CAMPPlus frame context and aligned reference prompts,
see the [reference-conditioning guide](../guides/reference-conditioning.md).

`CosyVoiceSpectrogramGenerator` uses a resolved
`quick_convert.components.features.mel.matcha.MelSpectrogramConfig` as its
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
  _target_: quick_convert.components.features.mel.matcha.MelSpectrogramConfig
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
Reconstruction training preserves separate content and mel clocks. A flow with
`token_mel_ratio=2` receives content lengths in content frames and target lengths
in mel frames. The wrapper accepts extraction-boundary differences of up to one
content frame, trims to complete aligned frames, and rejects larger duration
mismatches. Caller-owned features and lengths are preserved.

The W2V-BERT run configs explicitly load audio at the encoder's required sample
rate. Flow timing follows the encoder's `FRAME_HZ` capability; the `frame_ratio`
resolver requires a positive integer ratio to the configured 50 Hz mel clock.
Changing a model's capabilities or mel clock also requires compatible system
wiring, rather than rounding a noninteger ratio.

## Encoder and training behavior

The CFM reconstruction configs use the first-party
`quick_convert.components.sequence.upsample_conformer.UpsampleConformerEncoder`
adapter. The preset retains the original six-block stack, four-block second
stack, second projection, and three-frame lookahead. Upsampling follows the
content-to-mel ratio; factor one bypasses the resampler. See the
[encoder guide](upsample-conformer.md) for custom compositions and checkpoint
compatibility.

Frozen online encoders and a frozen vocoder remain in evaluation mode while
the reconstruction system trains. Encoders selected through
`trainable_online_encoders` retain their training mode and gradient path.
Frozen resource tensors can feed trainable downstream layers, including when a
backend computes them under `torch.inference_mode()`.

SSL reconstruction logs the learned layer-fusion weights and masked quantizer
usage every 1,000 training steps. Usage metrics include per-codebook utilization,
perplexity, and maximum usage fraction; padding and inactive codes are excluded.
Distributed counts are reduced in a fixed-size tensor before computing metrics.
Gradient-norm groups include feature fusion and any input projection. Set
`trainer.module.enable_grad_norm_logging=false` to disable gradient-norm logging.

The optional compile target is `system.decoder.flow`, so compilation replaces
the flow actually used by the system instead of creating a duplicate decoder
on the training module.

## Checkpoint compatibility

The resolved specification, pretrained architecture metadata, and model source
are saved as tensor-backed decoder state, so both Lightning checkpoints and
inference artifacts preserve them. Loading saved metadata does not require a
companion download. Explicit conflicting mel specifications are rejected.
Hydra-created vocoder configs are converted to independent plain lists and
dictionaries before serialization, allowing JSON metadata and weights-only
checkpoint restoration.

Checkpoints predating this metadata are rejected rather than silently resuming
with different targets. They learned the earlier 16 kHz / 8 kHz-upper-limit mel
representation. Migration must explicitly preserve that representation and use
an appropriate vocoder, or deliberately start training with the new feature
contract; simply assigning the new metadata does not repair old weights.

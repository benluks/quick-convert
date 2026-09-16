# Optional dependency profiles

This page records what each optional dependency group means and how confidently
it is supported. An installable extra is not automatically a supported workflow:
some remain available for compatibility or for experimental components that
still need architectural review.

## Classification

| Extra | Status | Purpose |
| --- | --- | --- |
| `manifests` | supported | Manifest and split operations backed by pandas. |
| `transformers` | supported | Hugging Face content encoders, including W2V-BERT and WavLM. |
| `training` | supported | Lightning training and the supported media loggers. |
| `sentencepiece` | supported | Tokenizer training and token decoding. |
| `wer` | supported | JiWER-based word-error-rate evaluation. |
| `cosyvoice` | supported | CosyVoice reconstruction flows, mel extraction, HiFT, and CAMPPlus. |
| `whisper` | integration | OpenAI Whisper ASR; retained but not part of the current reference training workflows. |
| `emotion2vec` | integration | FunASR-backed emotion2vec features. The package boundary is verified, but model download and inference remain optional integration tests. |
| `dac` | experimental | DAC content encoding. This is distinct from the deprecated DAC-style RVQ implementation. |
| `espnet-wavlm-joint` | integration | ESPnet WavLM speaker embeddings; file, tensor, batch, and configuration contracts are verified, while checkpoint-backed inference remains optional. |
| `pyannote` | integration | Pyannote WeSpeaker embeddings; file, tensor, batch, and configuration contracts are verified, while checkpoint-backed inference remains optional. |
| `mpm` | integration | Masked Prosody Model prosody features; adapter and local model execution are verified, while pretrained checkpoint loading remains an optional integration test. |
| `web` | legacy | Flask interface; currently unverified. |

Compatibility names remain available for existing environments:

- `w2vbert` aliases `transformers`;
- `lightning` aliases `training`;
- `asr` bundles `sentencepiece` and `wer`;
- `conditional-rvq` aliases `emotion2vec` under its former experimental name.

Speaker backend profiles are checked independently by the `Speaker integrations`
workflow. This keeps their large, mutually constrained dependency graphs out of
the fast core test environment while still verifying Python 3.11 installation
and backend imports.

The `Training integration` workflow similarly installs the `training` extra in
isolation and runs the Lightning adapters, logging, Hydra targets, and former
import paths without allowing dependency-based skips. The fast core workflow
therefore remains usable without Lightning.

## Verified conflicts

The current conflicts are dependency-level incompatibilities, not assumptions:

- `dac` cannot resolve with `espnet-wavlm-joint`: Descript Audio Tools requires
  Protobuf below 3.20, while S3PRL requires Protobuf 4.21.1 or newer.
- `mpm` cannot resolve with `espnet-wavlm-joint`: Masked Prosody Model requires
  NumPy 1.x, while ESPnet 202511 requires NumPy 2.x.

DAC and MPM can resolve together and therefore do not have a declared conflict.

## Open decisions

- [ ] Decide whether the DAC content encoder is still worth retaining; do not
  conflate this decision with retirement of the DAC-style RVQ layer.
- [x] Repair the Masked Prosody Model adapter and add a composable precompute configuration.
- [x] Bring the Pyannote and ESPnet speaker encoders under the current
  file/tensor/batch input and output contracts before calling them supported.
- [ ] Verify the Whisper and emotion2vec model-backed inference paths in optional
  integration jobs.
- [ ] Review or retire the legacy web interface.

### MPM pretrained checkpoint verification

The public `cdminix/masked_prosody_model` repository contains the two artifacts
expected by MPM 0.3.0: `model_config.yml` and `pytorch_model.bin`. The isolated
profile reaches that repository through `MaskedProsodyModel.from_pretrained()`.

The pretrained forward pass could not be completed in the cleanup workspace
because direct Hugging Face access is unavailable there. This is an environment
limitation, not evidence of a package or adapter failure. The same adapter was
executed end to end with the installed MPM preprocessing code and a locally
constructed MPM model; checkpoint-backed inference should still be verified in
a normally networked environment before promoting the integration to supported.

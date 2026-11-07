# PSST1 conversion

PSST1 generates a pseudo-speaker from the source ECAPA x-vector, compensates it
for emotion, and synthesizes 16 kHz audio from soft HuBERT units and YAAPT F0.
See the [system description](https://www.voiceprivacychallenge.org/vp2026/docs/T02_system_description.pdf)
for the research method.

## Use

Install the `speechbrain` and `yaapt` extras. The CLI's default emotion
recognizer also requires `speechbrain-ser`.

```bash
uv run anonymize psst1_iemocap dataset.root=/data/IEMOCAP_full_release
```

The run uses `SpeechConversionPipeline`. Set `target.seed=52` for reproducible
per-utterance sampling and `system.w=0.25 system.steps=10` for latent mixing and
Euler integration. Sampling hashes the seed and input waveform, so batch order
does not change assignments. Numerical results can differ across devices.

```python
from quick_convert.systems.conversion import GeneratedVoice, PSST1ConversionSystem

system = PSST1ConversionSystem()
target = system.prepare_target(GeneratedVoice(seed=52))
waveform = system.convert("utterance.wav", target=target, emotion="hap")
```

Use `SourceVoice()` instead of `GeneratedVoice()` for resynthesis with the
source speaker. For dataset runs, select
`target={_target_:quick_convert.systems.conversion.SourceVoice}`.

## Emotion labels

Labels are `neu`, `hap`, `sad`, and `ang`. An explicit `emotion=` argument
takes precedence over the loaded sample resource named by
`system.emotion_resource` (default `emotion`), followed by the configured
recognizer. Missing or unsupported labels raise errors. Use
`system.emotion_label_map` to translate other label sets.

The CLI defaults to `SpeechBrainSER` with
`speechbrain/emotion-recognition-wav2vec2-IEMOCAP`. Set
`system.emotion_recognizer=null` when providing dataset labels through a
[text resource provider](../data/index.md).

## Weights

Construction downloads about 465 MB from `QKTheoNguyen/psst1` at the pinned
`PSST1ConversionSystem.PRETRAINED_REVISION`. Pass `checkpoint_dir=` in Python
or `system.checkpoint_dir=/path/to/psst1` in the CLI to use local files.

| File | Contents |
| --- | --- |
| `ecapa.ckpt` | Pretrained SpeechBrain ECAPA-TDNN weights |
| `flow.pt` | Flow network and per-dimension normalization statistics |
| `emotion_boundaries.pt` | Emotion boundary normals |
| `hubert_soft.pt` | Converted HuBERT weights and soft-unit projection |
| `hifigan.pt` | HiFi-GAN weights with weight normalization folded in |

The contribution uses a retrained flow, Gaussian latent noise, and
per-dimension standardization to address issues reported in the research code.
Published research scores therefore need fresh evaluation for this release.
The investigation and parity evidence are recorded in [PR #52](https://github.com/benluks/quick-convert/pull/52).

## Train the speaker flow

Extract frozen ECAPA vectors, then train on LibriSpeech train-clean-360:

```bash
QUICK_CONVERT_PSST1_CHECKPOINT_DIR=/path/to/psst1 \
  uv run precompute spk_ecapa_librispeech data_root=/data
uv run train psst1_flow_librispeech data_root=/data
```

The recipe fits population statistics on training vectors and sets the global
seed before initialization. Its optimizer and schedule live in
`configs/trainer/psst1_flow.yaml` and `configs/trainer/optimizer/adamw_onecycle.yaml`.
It writes Lightning checkpoints and a research-compatible `flow.pt` export.
Checkpoints support the standard [inference and export APIs](../guides/inference.md).
Copy the exported `flow.pt` into your local PSST1 checkpoint directory to use it
for conversion.

## Implementation and provenance

`PSST1ConversionSystem` coordinates `SoftHubertContentEncoder`, YAAPT F0,
and pseudo-speaker generation. `PSST1HifiGanDecoder` accepts those features
and packs the conditioning required by its shared HiFi-GAN backbone. The PSST1 flow estimator lives under `components/decoders/flow_matching/estimators`;
speaker standardization and integration live in the speaker generator.

The estimator derives from the research `flow_matching/flow_matcher.py`
and facebookresearch/DiT. The HiFi-GAN adaptation derives from
facebookresearch/speech-resynthesis as modified by Xiaoxiao Miao (NII).
The contributor used a research working copy dated 2026-10-22; exact upstream
revisions were not recorded. The contribution states CC BY-NC 4.0 terms for
these adaptations and the release. Individual pretrained assets retain their
own license notices, listed on the release's model card.

To regenerate parity references, run `scripts/psst1/export_psst1.py OUT_DIR`
from the research repository's `gen/` directory in its Python 3.8/fairseq
environment. Set `QUICK_CONVERT_PSST1_EXPORT_DIR=OUT_DIR` when running
`tests/test_psst1_parity.py` with the `speechbrain` and `yaapt` extras.

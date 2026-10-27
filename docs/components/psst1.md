# PSST1 anonymizer

`PSST1Anonymizer` (`quick_convert.systems.anonymization`) replaces the speaker
of each utterance with a pseudo-speaker and compensates the pseudo-speaker for
the utterance's emotion. It returns 16 kHz audio of the same length as the
input. The system is described in [PSST1 System description for VoicePrivacy Challenge 2026](https://www.voiceprivacychallenge.org/vp2026/docs/T02_system_description.pdf)
(Nguyen, Luks, Bäckström and Martin, 2026).

Install the `speechbrain` and `yaapt` extras. The default emotion recognizer
also needs `speechbrain-ser`.

```python
from quick_convert.systems.anonymization import PSST1Anonymizer

anonymizer = PSST1Anonymizer(seed=52)
waveform = anonymizer.anonymize("utterance.wav", emotion="hap")
```

The first construction downloads the weights (about 465 MB) from the Hugging
Face Hub into the Hugging Face cache; later runs reuse them. Pass
`checkpoint_dir=` to use a local copy instead, for example on a machine
without internet access.

## Pipeline

For every utterance:

1. `SpeechBrainECAPASpeakerEncoder` computes a 192-d ECAPA-TDNN x-vector.
2. `PSST1FlowSpeakerGenerator` makes a pseudo-speaker. It standardizes the x-vector,
   inverts the flow to its Gaussian prior, blends the result with Gaussian
   noise, and runs the flow forward again. `w` is the weight of the original
   speaker (default 0.25) and `steps` is the number of Euler steps in each
   direction (default 10).
3. `EmotionBoundaryCompensator` rescales the pseudo x-vector to norm √192 and moves it to a signed distance of 40
   (−40 for `sad`) from the linear boundary of the utterance's emotion.
4. `PSST1Vocoder` resynthesizes the input's soft HuBERT units and
   YAAPT F0 with the compensated x-vector through HiFi-GAN.

`resynthesize()` skips steps 2 and 3, which gives the vocoder-only baseline.

## Emotion labels

PSST1 uses four labels: `neu`, `hap`, `sad`, `ang`. The label for an utterance
comes from the first available source:

1. the `emotion=` argument to `anonymize()`;
2. a sample resource named `emotion` (or `emotion_resource`) in dataset runs,
   for example from a `CSVAnnotationProvider` over an `utt2emo` file;
3. `emotion_recognizer.predict(waveform, sample_rate)`.

With no source, anonymization raises. `emotion_label_map` translates other
label sets, for example `{neutral: neu, happy: hap, excited: hap}`. A label that
is neither mapped nor one of the four raises an error naming the label.

The anonymizer config uses `SpeechBrainSER` with
`speechbrain/emotion-recognition-wav2vec2-IEMOCAP`, the model the research
code used for LibriSpeech. Replace or remove it with a Hydra override:

```bash
uv run anonymize psst1_iemocap anonymizer.emotion_recognizer=null
```

To read labels from a dataset instead, attach a text resource:

```yaml
dataset:
  resource_providers:
    - _target_: quick_convert.data.resources.CSVAnnotationProvider
      name: emotion
      path_template: /data/IEMOCAP_test/utt2emo
      delimiter: " "
```

## Reproducibility

Noise comes from a CPU `torch.Generator` seeded from `seed` and a hash of the
16 kHz input waveform. The same input and seed give the same output on every
device, different inputs get different pseudo-speakers, results do not depend
on batch order, and the global RNG is untouched. `seed=None` draws fresh noise
on every call.

## Differences from the research code

The port fixes two bugs in the research flow sampler, so its outputs do not
match the numbers in the [system description](https://www.voiceprivacychallenge.org/vp2026/docs/T02_system_description.pdf) exactly.

The research code blended in uniform noise (`torch.rand_like`), but the flow
was trained from Gaussian noise, and the blend only preserves the prior when
both inputs are Gaussian. The port uses `torch.randn`.

The research code also standardized every dimension with the first
dimension's mean and standard deviation (`zip` over the batch axis). The flow
was trained with per-dimension statistics, so the port standardizes each
dimension separately.

The vocoder, ECAPA encoder, flow network, and emotion compensation match the
research implementation stage by stage (`tests/test_psst1_parity.py`).

## Known issue: flow training data

The released `flow_301_epochs_enrolls` checkpoint was trained on train-clean-360
x-vectors extracted with an ECAPA-TDNN whose pretrained weights were never
loaded (the research extraction script comments out `load_collected`). Those
stored vectors match a randomly initialized ECAPA seeded with 1234 (cosine
1.0) and not the pretrained one (cosine about −0.13), and their per-dimension
scale is about 0.8 against about 10 for real x-vectors. The flow statistics
therefore do not describe the x-vectors this system feeds it. Retrain the flow
as described below and replace `flow.pt` in `checkpoint_dir`.

## Retraining the flow

As in the [system description](https://www.voiceprivacychallenge.org/vp2026/docs/T02_system_description.pdf), the flow models 192-dimensional x-vectors extracted from
LibriSpeech train-clean-360 with the frozen, pretrained ECAPA-TDNN, standardized
with the training set's empirical per-dimension mean and standard deviation.

1. Extract the x-vectors (extras: `speechbrain`). `ecapa.ckpt` is read from
   `QUICK_CONVERT_PSST1_CHECKPOINT_DIR`, a local checkpoint directory such as
   `hf download <repo> --local-dir /path/to/psst1`:

   ```bash
   QUICK_CONVERT_PSST1_CHECKPOINT_DIR=/path/to/psst1 \
     uv run precompute spk_ecapa_librispeech data_root=/data
   ```

   One `[192]` tensor per utterance is written under
   `outputs/precomputed/librispeech/ecapa/train-clean-360/`.

2. Train (extras: `training`):

   ```bash
   uv run train psst1_flow_librispeech data_root=/data
   ```

   `SpeakerGeneratorTrainingModule` first reads every training x-vector and
   calls `PSST1FlowSpeakerGenerator.fit_standardization` (population mean and
   standard deviation per dimension), then minimizes the conditional
   flow-matching loss. The recipe follows the research `flow_301` run: AdamW
   (learning rate 2e-4, betas 0.9/0.999, weight decay 0.01), OneCycleLR (10 %
   warm-up, cosine annealing, `div_factor` 25, `final_div_factor` 1000), batch
   size 128, 301 epochs, seed 1999. There is no validation split. Losses are
   logged locally to TensorBoard (`tensorboard --logdir outputs/psst1`).

3. When training ends, `flow.pt` is written to the run directory
   (`outputs/psst1/psst1_flow_train-clean-360/flow.pt`). Copy it into
   `checkpoint_dir`.

## Checkpoint directory

The released weights live in the Hugging Face repository
`PSST1Anonymizer.PRETRAINED_REPO_ID`, pinned to
`PSST1Anonymizer.PRETRAINED_REVISION`. `PSST1Anonymizer.download()` fetches
them and returns the cached directory; pass another `repo_id` or `revision` to
use a different release. A local `checkpoint_dir` holds the same five files:

| File | Contents |
|---|---|
| `ecapa.ckpt` | SpeechBrain ECAPA-TDNN state dict (VoxCeleb) |
| `flow.pt` | `velocity_net` state dict and per-dimension `mean`/`std` |
| `emotion_boundaries.pt` | unit boundary normals for `neu`, `hap`, `sad`, `ang` |
| `hubert_soft.pt` | torchaudio `hubert_base` state dict (`hubert`) and soft-unit projection (`W`), converted from fairseq |
| `hifigan.pt` | HiFi-GAN generator with weight normalization folded in |

The run config downloads the release by default; override it with
`anonymizer.checkpoint_dir=/path/to/psst1`.

`ecapa.ckpt`, `hubert_soft.pt`, `hifigan.pt` and `emotion_boundaries.pt` were
exported from the research repository by `scripts/psst1/export_psst1.py`;
`flow.pt` was retrained (see above). To regenerate the research assets and
parity references, run the export script in the research environment
(Python 3.8 with fairseq) from its `gen/` directory:

```bash
source env.sh
python /path/to/quick-convert/scripts/psst1/export_psst1.py /path/to/psst1_export
```

Then run the parity tests against the export:

```bash
QUICK_CONVERT_PSST1_EXPORT_DIR=/path/to/psst1_export uv run --extra speechbrain --extra yaapt pytest tests/test_psst1_parity.py
```

## Licensing

The release is for non-commercial use (CC BY-NC 4.0, following PSST). The
emotion boundaries are derived from ESD and MSP-Podcast. The ECAPA-TDNN,
HuBERT-soft and HiFi-GAN weights come from the VoicePrivacy 2024 baseline
models and keep their own licenses (CC BY 4.0 for VoxCeleb, LibriSpeech and
LibriTTS derivatives; MIT for HuBERT). The Hugging Face model card lists each
file's origin and license.

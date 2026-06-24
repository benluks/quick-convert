# DAC Content Encoder — Usage Guide

This change lets the anonymisation pipeline use a **Descript Audio Codec (DAC)**
encoder as its content front-end, in place of the W2V-BERT SSL encoder.

It adds two things:

- **`components/ssl/dac.py`** — `DACContentEncoder`, a drop-in content encoder
  that feeds the RVQ disentangler.
- **`pipelines/training/train_dac_from_scratch.py`** — a standalone script to
  train (or fine-tune) a DAC codec and produce encoder weights.

---

## 1. What it is

The pipeline turns a waveform into a frame-level "content" representation, then
disentangles it (speaker / content / emotion+prosody) with an RVQ stack. 

`DACContentEncoder` replaces that front-end with DAC's **single compact latent**:
- **50 Hz** frame rate (hop 320 at 16 kHz) → matches emotion2vec, the mel
  target, and the vocoder exactly, so **nothing downstream changes**.
- Output shape `(B, T, 1, D)` with `D = 1024`. The singleton "layer" axis lets
  the existing `ParallelConformerEncoder` consume it unchanged.

Only DAC's **encoder** is used. DAC's own quantizer and decoder are discarded —
the pipeline's RVQ and flow-matching decoder do those jobs.

---

## 2. Prerequisites

```bash
pip install descript-audio-codec   # brings in `dac` and `audiotools`
# torch + torchaudio are already required by the pipeline
```

A GPU is needed for any real training; CPU is fine only for the small test.

---

## 3. Three ways to get a DAC encoder

Pick based on objective of train at all.

### A. Use Descript's pretrained encoder (no training) — 

Nothing to run. Just build it in the pipeline:

```python
from quick_convert.components.ssl import DACContentEncoder

content_encoder = DACContentEncoder.from_pretrained("16khz")   # frozen
```

This downloads Descript's fully-trained 16 kHz codec and keeps its encoder. You
inherit all of their pretraining for free.

### B. Fine-tune from pretrained (adapt to your / VPC data)

Continue training Descript's codec on your audio, then load the resulting
encoder:

```bash
python pipelines/training/train_dac_from_scratch.py \
    --data /path/to/audio --init 16khz --lr 1e-5 --steps 50000 --save dac_ft.pt
```

### C. Train a codec from scratch (random init, no pretraining)

Only if you specifically need a from-zero codec (own architecture / data):

```bash
python pipelines/training/train_dac_from_scratch.py \
    --data /path/to/audio --init scratch --steps 250000 --save dac_scratch.pt
```

All three end the same way: a DAC encoder whose weights go into
`DACContentEncoder`. Only the *source* of the weights differs.

---

## 4. Wiring `DACContentEncoder` into the pipeline

Wherever config currently builds `W2VBertContentEncoder`, build the DAC
encoder there in it's place, and tell the conformer it now has **one** input layer:

```python
from quick_convert.components.ssl import DACContentEncoder
from quick_convert.components.encoders import ParallelConformerEncoder

# A) pretrained (frozen)                or  load your trained weights:
content_encoder = DACContentEncoder.from_pretrained("16khz")
# content_encoder = DACContentEncoder.from_scratch(trainable=True)
# content_encoder.dac_encoder.load_state_dict(torch.load("dac_ft_encoder.pt"))

conformer = ParallelConformerEncoder(
    input_dim=content_encoder.FEATURE_DIM,   # 1024 for DAC-16k  (was 1024 for W2V-BERT)
    num_layers=1,                            # was 24 for W2V-BERT's stacked layers
    # keep embed_dim and everything after it unchanged
)
```

**Frozen vs trainable:**

- `trainable=False` (default for `from_pretrained`) → the encoder is a fixed
  function, so you can **precompute features offline** as you do now (just much
  smaller). After swapping the encoder you must **regenerate the precomputed
  "content" features** — the old W2V-BERT ones wouldn't work here.
- `trainable=True` → the encoder stays in the autograd graph and trains jointly
  with the anonymiser. DAC is small enough to run **online**, which lets you skip
  the feature store entirely.

Nothing else changes: the RVQ, all distillation/adversarial heads, the speaker
teacher, the ASV/EER eval, the flow-matching decoder, and the vocoder are
untouched.

---

## 5. The training script in detail

`pipelines/training/train_dac_from_scratch.py` trains the **full** DAC codec
(encoder + RVQ + decoder) on audio reconstruction, then saves the weights. The
decoder is only a "teacher" during training — afterward you keep just the
encoder. 

### Arguments

| Flag | Default | Meaning |
|---|---|---|
| `--data` | `None` | Folder of `.wav`/`.flac` (searched recursively). Omit → synthetic smoke test. |
| `--init` | `scratch` | `scratch` = random init; a model_type (e.g. `16khz`) = fine-tune from pretrained. |
| `--steps` | `50000` | Training steps. |
| `--batch-size` | `16` | Batch size. |
| `--seconds` | `1.0` | Crop length per clip. |
| `--lr` | `1e-4` | Learning rate (use ~`1e-5` when fine-tuning). |
| `--sr` | `16000` | Sample rate (used only for `--init scratch`; otherwise taken from the checkpoint). |
| `--gan` | off | Add DAC's official discriminators (LS-GAN + feature matching). |
| `--save` | `dac16k_scratch.pt` | Output checkpoint path. |
| `--log-every` | `50` | Logging interval. |

### Losses

It uses DAC's **official** losses from `dac.nn.loss`
(`MelSpectrogramLoss` + `MultiScaleSTFTLoss` + waveform `L1Loss` + VQ
commitment/codebook, and `GANLoss` with `--gan`), combined with the official
`conf/base.yml` weights (mel 15, adv/feat 2, adv/gen 1, commitment 0.25,
codebook 1). As upstream, the stft/waveform terms are logged but not weighted in.

### Where to put data

Any folder of `.wav`/`.flac` files; pass its path to `--data`. For the
VoicePrivacy Challenge, point it at allowed training audio (e.g. a LibriSpeech
`train-*` root). Each clip is randomly cropped to `--seconds` at the working
sample rate.

### What to expect when it runs

```
init=16khz | hop=320 | frame_rate=50.0 Hz | latent=1024
step       0 | loss 12.431 | mel 0.812 | disc 1.998
step      50 | loss  9.107 | mel 0.563 | disc 1.742
...
saved full codec -> dac_ft.pt
saved encoder    -> dac_ft_encoder.pt
```

- The header prints the config — confirm `frame_rate=50.0 Hz` and `latent=1024`.
- `loss` should **trend downward** over time (`mel` is the main term).
- At the end it writes **two** files: the full codec (`--save`) and the
  encoder-only (`..._encoder.pt`), plus a snippet showing how to load the
  encoder into the pipeline.

With **no `--data`** it runs on synthetic noise — a "smoke test" that only
checks the loop executes end-to-end (build → forward/backward → save). The loss
won't meaningfully drop; it's not real training.

### Hardware / time

 training needs a GPU. From-scratch is long (the official recipe uses
~250k steps); fine-tuning (`--init 16khz`) converges much faster. Can use the
synthetic smoke test or `--steps 50` to verify the loop before a full run.

---

## 6. Handing the trained encoder to the pipeline

```python
import torch
from quick_convert.components.ssl import DACContentEncoder

enc = DACContentEncoder.from_scratch(trainable=True)          # empty architecture
enc.dac_encoder.load_state_dict(torch.load("dac_ft_encoder.pt"))  # fill with trained weights
```

`from_scratch` builds the matching empty `Encoder`; the saved `state_dict` fills
it. They line up because both construct the same DAC `Encoder`.

---

## 7. First-run checklist

- [ ] `pip install descript-audio-codec` succeeds (`dac` and `audiotools` import).
- [ ] `DACContentEncoder.from_pretrained("16khz")` downloads and builds; check
      `FEATURE_DIM == 1024` and `frame_hz == 50.0`.
- [ ] Set the conformer's `input_dim = FEATURE_DIM` and `num_layers = 1`.
- [ ] **Regenerate** precomputed "content" features (old W2V-BERT ones are stale).
- [ ] Smoke-test the trainer: `... --steps 50` (synthetic) or `--gan --steps 50`.
- [ ] Watch **WER** after the swap — content is the attribute a reconstruction
      codec exposes least cleanly; if it suffers, try fine-tuning the encoder
      (`trainable=True`) rather than freezing.

---

## 8. Gotchas

- **Sample rate.** DAC-16k needs 16 kHz input. Resample once at the dataset
  level (`target_sr=16000`, `load=True`); the encoder errors on a mismatch
  rather than silently resampling.
- **Frame rate is load-bearing.** Everything downstream assumes 50 Hz. DAC-16k
  gives exactly that; don't substitute a 24 kHz / 75 Hz codec without realigning
  the emotion2vec / mel / vocoder grid.
- **`from_pretrained` vs `from_scratch`.** Pretrained is frozen by default and
  is the safe baseline; `from_scratch` defaults to trainable and has no acoustic
  prior unless you load trained weights into it.

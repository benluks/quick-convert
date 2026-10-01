# Acoustic and paralinguistic content encoders

All adapters return `ContentFeatures` with batch first, time second, and valid
frame lengths. File inputs are converted to mono and resampled. Waveform and
`AudioBatch` inputs must already use the encoder's sample rate.

## SPEAR

Install `uv sync --extra spear`. The default is the official
[`marcoyang/spear-xlarge-speech-audio-v2`](https://huggingface.co/marcoyang/spear-xlarge-speech-audio-v2)
checkpoint: 16 kHz input, approximately 50 Hz output, 1280 features.

```python
from quick_convert.components.ssl import resolve_content_encoder

encoder = resolve_content_encoder("spear")(layer=None, device="cuda")
features = encoder.encode_file("speech.wav")
# values: (batch, frames, layers, features)
print(features.values.shape, features.lengths, features.frame_hz)
```

`layer=-1` (default) selects the final hidden state; `layer=0` selects the
first Zipformer layer. `layer=None` preserves all aligned intermediate states
for weighted-sum probes. `N_LAYERS` reports the actual checkpoint layer count
so ssl-probe can infer the fusion size automatically. The all-layer mode requires states with equal feature
dimensions; it does not interpolate heterogeneous states or pad feature channels.
The adapter supports v2 checkpoints with `output_downsampling_factor=1` and
reads feature dimensions from their configuration. It rejects older checkpoints
with a different output timebase. `revision` can pin the upstream release;
`local_files_only=True` uses cached weights and code. The official model needs
`trust_remote_code=True`, which is explicit in the constructor.

Hydra component: `components/ssl=spear`. Configure the consuming system's feature
dimension to match the chosen checkpoint; changing an encoder does not retrain
an existing decoder.

## PASE / PASE+

Install `uv sync --extra pase`, and obtain the frontend configuration and encoder
checkpoint from the [official PASE repository](https://github.com/santi-pdp/pase).
For the published PASE+ model these are `cfg/frontend/PASE+.cfg` and `FE_e199.ckpt`.
The adapter requires local paths and never substitutes random weights for a
missing checkpoint. It accepts plain state dictionaries or a `state_dict` wrapper,
with strict loading.

```python
encoder = resolve_content_encoder("paseplus")(config_path="PASE+.cfg", checkpoint_path="FE_e199.ckpt", device="cuda")
features = encoder.encode_file("speech.wav")
```

PASE+ produces 256-dimensional features at 100 Hz for its published configuration.
`pase` and `paseplus` resolve to the same adapter; the config/checkpoint determines
the actual model. The default `layer=-1` returns the final frontend representation.
With `layer=None`, dense-sum frontends expose their pretrained projected skip
stages followed by that final aggregate output; nonnegative indices select a
stage. The published PASE+ config gives seven skip stages plus one final output,
all 256-dimensional at 100 Hz. Alignment reuses upstream crop/mean downsampling;
no randomly initialized projections are introduced. These are projected CNN
contributions plus an aggregate, not eight sequential transformer states.
Frontends without compatible dense-sum skips reject intermediate extraction.
Waveforms are encoded separately before padding because its recurrent frontend
and temporal normalization depend on each utterance's true extent.

The published config additionally needs the upstream QRNN/CuPy environment
(`torchqrnn` from [pytorch-qrnn](https://github.com/salesforce/pytorch-qrnn)).
Those legacy, CUDA-specific dependencies are not installed by the base PASE extra;
follow the upstream setup for your CUDA version. Do not change `rnn_type` to bypass
that requirement with the published checkpoint: that changes the architecture.
This backend has not been verified with the published checkpoint in CI.

Hydra component: `components/ssl=paseplus`; supply `config_path` and
`checkpoint_path`, both intentionally required.

## emotion2vec

This encoder already exists: `resolve_content_encoder("emotion2vec")` or
`components/ssl=emo2vec`. Install `uv sync --extra emotion2vec`.
The extra constrains Transformers to `>=4.50.1,<5`, preventing resolution to
legacy versions whose tokenizers require a source build on Python 3.11.
To install the latest GitHub code into an existing environment, use:

```bash
uv pip install --python .venv/bin/python --upgrade \
  'quick-convert[emotion2vec] @ git+https://github.com/benluks/quick-convert.git@main'
```

Merging a pull request updates GitHub; package-index releases update separately.

The default `iic/emotion2vec_plus_large` uses 16 kHz audio and returns final-layer
frame embeddings at approximately 50 Hz. `granularity="utterance"` returns one
pooled vector and `frame_hz=None`. Use `layer=None` to return all shared transformer
block states for weighted-sum probing, or a nonnegative index to select a block.
The adapter calls the underlying model with masking disabled, preserves its
per-utterance waveform normalization, removes auxiliary tokens, and captures
residual block states rather than pretraining-target tensors. The last returned
state includes the final model normalization and matches the final frame output.
The audio frontend blocks are excluded. `N_LAYERS` reports the checkpoint-specific
shared transformer depth. Intermediate extraction requires frame granularity.
For local weights,
pass a local model directory as `model_name`; `local_files_only=True` is rejected
because this adapter cannot enforce it through FunASR.

## VoiceFM-Whisper

Install `uv sync --extra voicefm`. The [official VoiceFM repository](https://github.com/oelemento/VoiceFM-public)
contains code only; clinically trained weights are distributed separately.
Obtain a VoiceFM-Whisper checkpoint from the authors or train one first.
This adapter requires a local checkpoint and never substitutes base Whisper weights.

```python
encoder = resolve_content_encoder("voicefm")(checkpoint_path="voicefm_best_model.pt", layer=None, device="cuda")
features = encoder.encode_file("speech.wav")
```

The adapter exposes the clinically fine-tuned **Whisper encoder before pooling**:
1280-dimensional frames at 50 Hz for Whisper large-v2, or all 32 transformer
layers with `layer=None`. `N_LAYERS` lets ssl-probe infer the weighted-sum size.
`layer=-1` selects the final state; `layer=0` selects the input embedding.
These are not the paper's normalized, pooled 256-dimensional clinical embeddings:
the task embedding and projection head are not applied to individual frames.
The HuBERT and HeAR variants are outside this adapter.

Supported checkpoint forms are an upstream training checkpoint with
`model_state_dict` keys beginning `audio_encoder.encoder.`, an audio-encoder
state dictionary with `encoder.` keys, or a bare Whisper encoder state dictionary.
A `state_dict` wrapper is also accepted. All encoder weights must be present and
match the configured architecture; incomplete or mismatched checkpoints fail
strictly. Clinical and projection weights are excluded from frame extraction.
Only the Whisper config and frontend are fetched from `model_name`, rather than
loading a full base encoder/decoder checkpoint. For offline extraction, cache
those assets or use a local model config/frontend directory with
`local_files_only=True`.

The input timebase is preserved: no silence trimming or peak normalization is
performed. Set such preprocessing at the dataset level if desired. Whisper pads
each utterance to its 30-second context; returned values are cropped to the
longest valid frame extent with individual lengths computed from the frontend
mask and convolution geometry. Its attention still includes the padded context,
as in the upstream encoder. Inputs exceeding 30 seconds are rejected rather
than silently truncated; split long recordings before extraction.

Hydra component: `components/ssl=voicefm`, with required `checkpoint_path`.
Verification covers checkpoint round trips and length/layer contracts; published
clinical VoiceFM weights have not been exercised because they are not available
in the code release.

## CARE: integration prerequisite

The [official CARE repository](https://github.com/iiscleap/CARE) contains training
and downstream feature-extraction code, but its example refers to local training
checkpoints and does not provide a downloadable pretrained checkpoint. CARE is
therefore not registered as an available pretrained encoder. Integration needs an
actual CARE checkpoint, its architecture options (including convolutional
branches), and a choice of acoustic, semantic, or concatenated representations.
Using an ordinary WavLM checkpoint under a CARE alias would not reproduce CARE.

HeAR and TRILLsson are also outside this addition: their released pooled interfaces
are not interchangeable with frame-level speech encoders.

## Weighted-sum examples

```python
emotion = resolve_content_encoder("emotion2vec")(layer=None, device="cuda")
pase = resolve_content_encoder("paseplus")(
    config_path="PASE+.cfg", checkpoint_path="FE_e199.ckpt", layer=None, device="cuda"
)
# Both return (batch, frames, layers, features), with encoder.N_LAYERS.
```

Layer-wise backend access is checked by fast contract tests. Published emotion2vec
and PASE+ checkpoint inference remains an additional model smoke test, particularly
for PASE+'s legacy QRNN/CuPy setup.

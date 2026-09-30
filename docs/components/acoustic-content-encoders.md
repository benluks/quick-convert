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
for weighted-sum probes. The all-layer mode requires states with equal feature
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
the actual model. Only the final frontend representation is exposed.
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

The default `iic/emotion2vec_plus_large` uses 16 kHz audio and returns final-layer
frame embeddings at approximately 50 Hz. `granularity="utterance"` returns one
pooled vector and `frame_hz=None`. FunASR's public extraction interface does not
select intermediate layers: only `layer=-1` is accepted. For local weights,
pass a local model directory as `model_name`; `local_files_only=True` is rejected
because this adapter cannot enforce it through FunASR.

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

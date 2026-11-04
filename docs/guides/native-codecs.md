# Native codec representations

The codec adapters expose frozen native quantizer contributions as floating-point
vectors. They do not retrain a quantizer or return integer token IDs.

| Encoder | Optional extra | Output |
|---|---|---|
| `DACQuantizedContentEncoder` | `dac` | `[batch, time, codebooks, dimension]` |
| `MimiQuantizedContentEncoder` | `mimi` | `[batch, time, codebooks, dimension]` |
| `SpeechTokenizerQuantizedContentEncoder` | `speechtokenizer` | All contributions, or one selected layer |

```python
from quick_convert.components.ssl import SpeechTokenizerQuantizedContentEncoder

encoder = SpeechTokenizerQuantizedContentEncoder(layer=0, device="cpu")
features = encoder.encode_waveforms(waveforms, lengths, sample_rate=16000)
# features.values: [batch, padded_frames, dimension]
# features.lengths: valid frames per item
```

SpeechTokenizer downloads its default `speechtokenizer_hubert_avg` configuration
and weights on construction. To use local files, supply both `config_path` and
`checkpoint_path`. `layer=None` returns all contributions; a zero-based index
returns a single `[batch, time, dimension]` representation. This is the native
quantizer contribution, before SpeechTokenizer's optional semantic projection.

DAC defaults to its 16 kHz model. Mimi defaults to `kyutai/mimi`; query each
instance's `sample_rate`, `frame_hz`, `feature_dim`, and `n_codebooks` for loaded
model capabilities. Mimi requires Transformers 4.57.1 or newer. Importing the
adapter classes does not import their optional backend packages or download
models; construction does.

Each valid waveform is encoded separately, excluding batch padding. Outputs
report actual returned frame counts, are detached, and can feed a trainable
downstream module. Input tensors are preserved, and codec modules stay in
evaluation mode when a parent enters training. `encode_file` mixes channels to
mono and resamples; waveform/batch methods require the native sample rate.
All-codebook outputs need an explicit downstream reduction or layer selection
before use by a decoder expecting three-dimensional content.

## Supported workflows

```bash
quick-convert requirements precompute_content_speechtokenizer_librispeech
quick-convert precompute_content_speechtokenizer_librispeech device=cuda
quick-convert train_sslr_speechtokenizer_dit_campp_prefix device=cuda
```

The precompute root selects layer zero. Reconstruction roots
`train_sslr_speechtokenizer_cmdiff_campp_prefix` and
`train_sslr_speechtokenizer_dit_campp_prefix` use that layer with a CAMPPlus
reference stream. Their 16 kHz, 50 Hz, 1024-dimensional wiring targets the default
checkpoint; custom checkpoints may require composition overrides. Configure
dataset roots and manifests as in the [workflow guide](workflows.md), and inspect
requirements with the same overrides before installing extras.

Backend contracts follow the upstream implementations:
[DAC](https://github.com/descriptinc/descript-audio-codec/blob/main/dac/nn/quantize.py),
[Mimi](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/mimi/modeling_mimi.py),
and [SpeechTokenizer](https://github.com/ZhangXInFD/SpeechTokenizer/blob/main/speechtokenizer/model.py).

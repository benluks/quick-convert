# Paired reconstruction and frozen VQ-ASR content

Paired reconstruction extracts linguistic content from synthesized speech while
predicting the original recording's mel spectrogram. A separate same-speaker
recording supplies a reference prefix. The three audio roles are independent:
content audio, target audio, and speaker/reference audio.

## Paired TTS training

The supported roots are `train_sslr_tts_dit_campp_prefix`,
`train_sslr_tts_dit_mel_prefix`, and `train_sslr_tts_dac_dit_mel_prefix`.
The first two use W2V-BERT's final layer; the third uses continuous DAC features.
They train a first-party Conformer content encoder and DiT decoder. The content
sequence is resized per example to the target mel duration, so synthesized and
original recordings need not have equal lengths.

Prepare `speaker_split/train_tts_checked.csv` and
`speaker_split/val_tts_checked.csv` under `dataset_path`. Each manifest needs
`utt_id`, `path`, `split`, and `spkid`. Include only verified pairs and use
speaker-disjoint train/validation splits. Each speaker needs another recording
within its split for the reference provider. The presets do not generate TTS
audio or check pair quality for you.

For a source path ending in
`LibriSpeech/train-clean-100/19/198/19-198-0000.flac`, place synthesized audio at
`<tts_data_path>/train-clean-100/19/198/19-198-0000.flac`. The `tts_path` resource
uses this mirrored layout. Override the provider for other layouts.

```bash
quick-convert requirements train_sslr_tts_dac_dit_mel_prefix
quick-convert train_sslr_tts_dac_dit_mel_prefix \
  dataset_path=/data/paired tts_data_path=/data/tts device=cuda
```

`reference_seconds` limits independently loaded reference audio. Training scores
the entire original target after prepending reference context; reference frames
do not contribute to the target loss. Missing pairs, empty recordings, and a
reference/content path resolving to the target itself raise errors. Original
batch audio and resources are preserved.

For duration-controlled inference, pass a positive scalar or per-example tensor
as `duration_ratio` to `system.generate_features(..., run_vocoder=False)`.
The ratio multiplies the source's nominal mel length. Returned mel lengths report
the requested duration. Other reconstruction decoders reject non-unit ratios.

## Frozen VQ-ASR bottleneck

`VQASRContentEncoder` exposes projected quantized vectors from a restored public
`VQASRSystem`, without its CTC head. It freezes the supplied encoder, layer fusion
and quantizer in place, including EMA state; load a separate source system if it
must remain trainable elsewhere.

```python
from quick_convert.components.ssl import VQASRContentEncoder
from quick_convert.inference import load_inference_system

source = load_inference_system("/models/vq-asr-export", map_location="cpu")
content_encoder = VQASRContentEncoder(source)
features = content_encoder(batch)
```

`train_sslr_vq_asr_dit vq_asr_checkpoint=/models/vq-asr-export` composes this
adapter with reconstruction training. `vq_asr_config` optionally selects a saved
source configuration. This recipe assumes a 1024-dimensional, 50 Hz bottleneck
and 16 kHz input; override decoder dimensions, frame rate and dataset sample rates
for other sources. The source must include an online content encoder. Saved
reconstruction configurations retain the nested source location, which must be
available when reconstructing the system for loading or export. The requirement
planner inspects that source's saved composition without loading its weights.

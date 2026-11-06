# Supported workflows

Run `quick-convert --help` to list the composition roots installed with your version. The universal form accepts the complete run name; verb-oriented aliases add their prefix automatically.

| Goal | Required profile | Canonical command | Main output |
|---|---|---|---|
| Build a flat LibriSpeech manifest | base | `quick-convert build_flat_manifest_libri data_root=/data` | `outputs/librispeech/manifest.csv` |
| Train the reference tokenizer | `sentencepiece`, `training` | `train bpe_tokenizer_librispeech data_root=/data` | tokenizer model under `outputs/tokenizer/` |
| Precompute token IDs | `sentencepiece` | `precompute tokens_librispeech data_root=/data` | token sidecar tensors |
| Precompute W2V-BERT content | `transformers` | `precompute content_w2vbert_librispeech data_root=/data` | content sidecar tensors |
| Train VQ-ASR | `transformers`, `asr`, `training` | `train vq_asr_librispeech data_root=/data` | resolved config and Lightning checkpoints |
| Train CosyVoice reconstruction | `transformers`, `cosyvoice`, `training` | `train sslr_w2vbert_cmdiff data_root=/data` | resolved config and Lightning checkpoints |
| Evaluate Whisper ASR | `whisper`, `wer` | `evaluate asr_librispeech data_root=/data` | evaluation results beneath the configured output directory |
| Anonymize CLAC with KNN-VC | backend-specific dependencies | `anonymize knnvc_clac` | converted audio beneath the configured output directory |
| Export a training run | base plus classes used by the system | `quick-convert export RUN_DIR DESTINATION` | versioned inference artifact |

Commands shown without `uv run` assume an activated environment. In a development checkout, prefix them with `uv run`.

## Reproducible and distributed runs

Set a top-level seed before constructing datasets and models:

```bash
quick-convert train_vq_asr_librispeech +seed=17
```

The runner seeds Python, NumPy, and PyTorch before pipeline instantiation. The
seed must be an integer in `[0, 2**32)`. This makes random initialization
repeatable. See [distributed training and compilation](training.md#distributed-training-and-compilation)
for GPU execution settings.

## Hydra overrides

Append overrides after the run name:

```bash
uv run train vq_asr_librispeech \
    data_root=/data \
    device=cuda \
    trainer.train_dataloader_kwargs.batch_size=16
```

Inspect the run YAML before launching an expensive job. A composed configuration can be syntactically valid while still requiring local data, credentials, cache space, or a large model download.

## End-to-end reference

The [VQ-ASR quickstart](../quickstart.md) covers tokenizer training, token ID
precomputation, manifest preparation, training, and inference export in order.

## Speaker verification

See [automatic speaker verification](asv.md) for enrollment/test manifests,
public speaker encoders, cosine scoring, and EER reporting.

## Fit reusable transformations

Use `fit_speaker_index_librispeech` to persist a label mapping, or precompute
openSMILE sidecars and run `fit_prosody_librispeech` for acoustic statistics.
See the [fitting guide](fitting.md) for resource contracts and saved-state reuse.

## Resume feature precomputation

See [feature precomputation](precomputation.md#resume-feature-precomputation)
for cache identity, interruption behavior, and legacy caches.

## Train and validate a system

See [training runs](training.md) for DDP, compilation, configuration logging,
and the pretrained encoder gradient check. For SSL reconstruction, start with
[reference conditioning](reference-conditioning.md) or
[paired reconstruction](paired-reconstruction.md).

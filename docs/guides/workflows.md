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

The [VQ-ASR quickstart](../quickstart.md) covers tokenizer training, feature precomputation, manifest preparation, training, and inference export in order.

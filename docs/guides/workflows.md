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

The [VQ-ASR quickstart](../quickstart.md) covers tokenizer training, token ID
precomputation, manifest preparation, training, and inference export in order.

## Speaker verification

See [automatic speaker verification](asv.md) for enrollment/test manifests,
public speaker encoders, cosine scoring, and EER reporting.

### W&B training configuration

Training through `uv run train ...`, `quick-convert train_...`, or
`python -m quick_convert.cli.train` records the full resolved Hydra configuration
in W&B when the Lightning trainer has a W&B logger. This happens after the local
`config.yaml` is written and before fitting. The file is also uploaded as
`config.yaml` in the run's files. Interpolations are resolved to their effective
values, so settings are available for run comparison and filtering.

Only rank zero logs this configuration in distributed training. Backends without
configuration logging and runs without W&B still save their local configuration.
VQ-ASR keeps instantiated systems, encoders, and optimization objects out of its
checkpoint hyperparameters and disables automatic hyperparameter logging; the
resolved Hydra config supplies the W&B settings instead. Avoid putting secrets
in configuration fields, since the full resolved configuration is logged.

## Pretrained encoder gradient smoke test

From a repository checkout, run a short audio clip through pretrained WavLM
and the system's online resource path:

```bash
uv run --extra transformers python scripts/check_wavlm_gradients.py \
    /path/to/short.wav --device cuda
```

The script uses the last hidden layer and a tiny regression head with a synthetic
MSE target. It prints `PASS frozen` and `PASS trainable` after checking finite,
nonzero encoder gradients, encoder weight changes, and head updates. Frozen
encoder weights must remain unchanged. It runs in evaluation mode to avoid
stochastic masking/dropout while retaining autograd, and saves no checkpoints.
Use `--device cpu` without a GPU, `--model-name /path/to/model` for a local model,
or `--local-files-only` to require cached weights. The default downloads
`microsoft/wavlm-large`; full backward needs substantially more memory than inference.
This checks gradient plumbing, not jitter prediction or preservation of SSL utility.

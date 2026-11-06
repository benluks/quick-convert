# Supported workflows

For reconstruction with reference prefixes, see the
[reference-prompt and conditioning-stream workflows](reference-conditioning.md).
Start with `train_sslr_w2vbert_dit_streams`, or choose a CAMPPlus-prefix preset
for frame-wise speaker context.

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
repeatable; it does not enable deterministic GPU algorithms.

For two visible CUDA GPUs, enable DDP through the normal console command:

```bash
quick-convert train_vq_asr_librispeech \
  trainer.ddp.enabled=true \
  trainer.train_dataloader_kwargs.batch_size=8 \
  +trainer.trainer_kwargs.accumulate_grad_batches=4
```

Batch size is per GPU, giving an effective batch of 64 in this example. The
entry point preserves its executable path when Lightning relaunches ranks.
Only rank zero initializes the logger and writes the run configuration; other
ranks use the output root locally until distributed setup finishes.

`trainer.compile.targets` accepts dotted module paths such as `[system.decoder]`
as well as direct module attributes. Each selected module is replaced in place
with its compiled version before training.

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

From a repository checkout, run pretrained WavLM through the system's online
resource path. No audio file is required:

```bash
uv run --extra transformers python scripts/check_wavlm_gradients.py \
    --device cuda
```

By default it generates one second of a tone plus noise. Optionally pass an
audio path; the waveform is trimmed to the first second after resampling.
`--seconds 2` changes the duration (allowed range: 0.1–10 seconds). Synthetic audio
is sufficient for this gradient check; it does not measure speech performance.

The script uses the last hidden layer and a tiny regression head with a synthetic
MSE target. It prints `PASS frozen` and `PASS trainable` after checking finite,
nonzero encoder gradients, encoder weight changes, and head updates. Frozen
encoder weights must remain unchanged. It runs in evaluation mode to avoid
stochastic masking/dropout while retaining autograd, and saves no checkpoints.
Use `--device cpu` without a GPU, `--model-name /path/to/model` for a local model,
or `--local-files-only` to require cached weights. The default downloads
`microsoft/wavlm-large`; full backward needs substantially more memory than inference.
This checks gradient plumbing, not jitter prediction or preservation of SSL utility.

### Fit reusable transformations

Use `fit_speaker_index_librispeech` to persist a label mapping, or precompute
openSMILE sidecars and run `fit_prosody_librispeech` for acoustic statistics.
See the [fitting guide](fitting.md) for resource contracts and saved-state reuse.

## Resume feature precomputation

With `pipeline.skip_existing=true`, the precompute workflow loads and extracts
only recordings whose feature file is missing. It publishes complete feature
files individually and replaces `manifest.jsonl` only after the run succeeds.
The manifest includes both retained and newly computed features.

Keep the same dataset rows, extractor configuration, and input revision when
resuming. The output directory's `provenance.json` records the resolved dataset
and extractor configuration plus a digest of ordered sample IDs, splits, and
paths. Batch size and worker count can change between runs. Use
`+pipeline.cache_key=my-input-revision` to identify changes that the config and
paths do not capture, such as replacing weights or audio at the same path.
Python callers enabling resume must pass `cache_key="my-input-revision"`.

For an older output directory without an identity, or when changing the
extractor or input revision, set `pipeline.out_dir` to a fresh directory. The
feature paths within that directory and the saved tensor format are unchanged.

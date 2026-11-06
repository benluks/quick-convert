# Training runs

Start with the [VQ-ASR quickstart](../quickstart.md) or choose an
[SSL reconstruction preset](reference-conditioning.md). This guide covers
execution settings shared by training runs.

## Distributed training and compilation

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

## W&B training configuration

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
Use this check when validating gradient flow through an online encoder.

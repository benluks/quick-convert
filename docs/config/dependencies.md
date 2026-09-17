# Optional dependency planning

Quick Convert derives optional installation requirements from the fully
composed Hydra object graph. This keeps environment selection aligned with
component selection: replacing W2V-BERT with DAC changes the required backend
without requiring a separate workflow extra.

Inspect a run before installing its optional dependencies:

```bash
uv run quick-convert requirements train_vq_asr_librispeech
```

The command reports semantic extras and prints installation commands for a
checkout and for a published package. Hydra overrides participate in planning:

```bash
uv run quick-convert requirements train_vq_asr_librispeech \
  system.online_encoders.content._target_=quick_convert.components.ssl.DACContentEncoder
```

Check the active environment with:

```bash
uv run quick-convert doctor train_vq_asr_librispeech
```

Use `--json` with either command for tools and automated agents.

## Ownership

- `pyproject.toml` maps extras to installable Python distributions.
- `quick_convert/configs/dependencies.yaml` maps configured targets to those
  extras and defines lightweight import probes.
- Hydra run and component configs select the object graph.

The dependency registry is metadata for planning only. It is deliberately kept
outside instantiation configs, so dependency declarations are never forwarded
as constructor arguments.

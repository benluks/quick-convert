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
  system.online_encoders.content._target_=quick_convert.components.encoders.content.DACContentEncoder
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

## Logging and ONNX compatibility

The `training`, `lightning`, and `dac` extras select `wandb==0.25.0` and
`protobuf<7`. Keep these supported constraints aligned when combining logging
and codec profiles.

The `s3tokenizer` extra requires `ml-dtypes>=0.5.0` for its ONNX imports.
It cannot share a uv environment with `mpm`, whose TensorFlow dependency
requires an older `ml-dtypes`; install those profiles separately.

## Prefix-conditioned reconstruction

`campplus` supplies `onnx`, `onnxruntime` and `huggingface-hub` for the
pooled and frame-wise CAMPPlus encoder. Imports and model downloads are lazy until that
encoder is selected. Its supported ONNX graph is truncated in memory and is
never written back to the model cache.

The DAC profile pins AudioTools to an upstream revision with a broader Protobuf
range, allowing DAC and CAMPPlus/ONNX context in the same environment. The
[pinned upstream requirements](https://github.com/descriptinc/audiotools/blob/348ebf2034ce24e2a91a553e3171cb00c0c71678/setup.py#L55)
are retained through a direct VCS requirement; selecting this profile needs Git
and access to that public repository. Existing DAC/ESPnet profile conflicts are
unchanged.

Run `quick-convert requirements train_sslr_dac_dit_campp_prefix` to inspect the
complete profile, or choose the W2V-BERT prefix variants when DAC is not needed.
The same-speaker reference resource provider selects the existing `manifests`
profile; prompt assembly and the stream-packing components need only the base
dependencies.

## Profiles, compatibility, and verification

`pyproject.toml` is the complete list of installable extras. `asr` bundles
`sentencepiece` and `wer`; `w2vbert`, `lightning`, and `conditional-rvq` retain
compatibility names for `transformers`, `training`, and `emotion2vec`.
The declared uv conflicts include DAC/ESPnet, MPM/ESPnet, and MPM/S3Tokenizer;
inspect the current file rather than maintaining a separate package matrix.

A resolved installation plan verifies dependency metadata, not pretrained
model behavior. Optional [acoustic](https://github.com/benluks/quick-convert/blob/main/.github/workflows/acoustic-model-smoke.yml),
[speaker](https://github.com/benluks/quick-convert/blob/main/.github/workflows/speaker-integrations.yml), and
[training](https://github.com/benluks/quick-convert/blob/main/.github/workflows/training-integration.yml) jobs exercise their
own backend profiles. Pretrained MPM inference still needs checkpoint-backed
verification; locally constructed-model tests are a different boundary.
Remaining backend and legacy-interface work is tracked in the
[architecture backlog](../maintenance/architecture_backlog.md).

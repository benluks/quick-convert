# Extending Quick Convert

Choose the narrowest role that owns the new behavior.

## Add a component

Use a component for a reusable encoder, decoder, loss, feature extractor, layer, or signal-processing operation.

- Define explicit tensor shapes and valid-length behavior.
- Keep data iteration and persistence outside the component.
- Load optional backends lazily.
- Add a focused unit test with synthetic tensors.
- Add a Hydra config only when declarative construction is useful.

## Add a system

Use a system for a complete task-level inference capability.

- Compose components behind a stable inference method.
- Return a typed result when multiple outputs or lengths are meaningful.
- Do not import Lightning or a pipeline.
- Keep training losses out unless they are intrinsic to the task result.
- Verify that the system can be saved and loaded as an inference artifact.

Subclass `quick_convert.systems.base.InferenceSystem` to inherit typed
`from_run` and `from_artifact` loaders. Use top-level `system` in the run config
and `self.system` in its training wrapper. New checkpoints then embed inference
composition through the standard training pipeline. No per-model loader is
needed for this contract; any configured `torch.nn.Module` also works with
`load_inference_system`. See the [inference guide](../guides/inference.md).

## Add training behavior

Wrap a system in a training module when adding objectives, logging, or optimization-specific behavior.

- Keep the wrapped system accessible as `system`.
- Store system weights beneath the `system.*` checkpoint prefix.
- Explicitly exclude reproducible external state when appropriate.
- Keep exact-resumption checkpoints distinct from exported inference artifacts.

## Add a pipeline

Use a pipeline when coordinating datasets, output directories, persistence, or a multi-step workflow.

- Construction should not start expensive work.
- Make preparation explicit and idempotent where practical.
- Let systems and components own task computation.
- Write the resolved config beside persisted outputs.

## Add a dataset resource

Prefer a named resource provider over extending `AudioSample` for experiment-specific metadata.

- Choose a stable resource name and kind.
- Keep location policy in the provider.
- Let dataset `load` policy control materialization.
- Define collation behavior for new value types.

### Resolve resources inside a system

Systems using `OnlineResourceMixin.get_resource(batch, name)` receive a
`ResolvedResource` with `values`, optional `lengths`, and optional `metadata`.
Loaded resources take precedence over online encoders. Online encoders are
frozen by default; only explicitly selected trainable encoders retain gradients.
Loaded tensor resources are detached even when an online encoder is trainable.

`ContentFeatures` results preserve their representation descriptors in metadata:
`feature_dim`, `representation_type`, `temporal_granularity`, `backend`,
`model_name`, `layer`, and `frame_hz`. This describes the source representation;
consumers must account for later transforms separately. `TensorResourceBatch`
and plain tensors have no representation metadata, and resolution does not
invent it. A `None` field such as `frame_hz` remains `None`.

For an existing system and batch:

```python
content = system.get_resource(batch, "content")
values, lengths = content.values, content.lengths
frame_hz = content.metadata.get("frame_hz") if content.metadata is not None else None
```

Custom encoders may return `ResolvedResource(values, lengths, metadata=...)`
to supply their own metadata mapping. Resolution copies the mapping so edits
to its keys do not mutate the source; nested custom values are shared.
Legacy `(values, lengths)` tuples and wrappers with a non-callable `.values`
attribute and optional `.lengths` remain supported, without metadata inference.
Other objects, including dictionaries, remain ordinary resource values.

### Adapt content representations

Reconstruction and VQ-ASR share the mechanics for applying feature transforms,
calculating valid lengths, and invoking RVQ. Systems retain ownership of their
modules and task-specific output selection: reconstruction uses `z_q`, while
ASR may use latents and apply its post-quantization network and CTC head.

Content inputs are floating-point `[batch, time, feature]` tensors, optionally
with a layer axis before fusion. Transforms must produce three-dimensional
features and implement `output_lengths(input_lengths)`; `nn.Identity` preserves
lengths automatically. Nonempty model sequences require positive integer
lengths that fit their storage. RVQ uses `[batch, feature, time]` internally;
systems expose batch/time/feature layout again afterward.

Existing tensor APIs still work. To check source metadata explicitly:

```python
content = system.get_resource(batch, "content")
result = system.encode_features(content.values, content.lengths, metadata=content.metadata)
```

Batch-based entry points and reconstruction training pass this metadata
automatically. Declared feature dimensions must match source storage; these
systems accept frame-level continuous or quantized vectors, not token IDs.
The source mapping is never rewritten as if it described transformed features.

`PRESERVES_FRAME_RATE = True` explicitly declares unchanged temporal spacing
for a module. Identity, layer fusion, the first-party time-preserving Conformers,
and RVQ retain a known source rate. Through other transforms the rate becomes
unknown; output lengths alone do not establish a clock. Reconstruction checks
a retained rate against the decoder's declared input rate. Precomputed tensors
without metadata remain supported without guessing a source rate.

CosyVoice training and generation validate the same content shape, feature
dimension, and length rules. Generation trims excess padding but no longer
silently shortens lengths that exceed storage. Such inputs now raise an error;
callers must supply accurate lengths. Empty conditioning prefixes retain their
separate contract. Module state-dict paths and existing result types are unchanged.

## Add an optional backend

- Put the dependency in a meaningful optional profile.
- Raise an actionable import error at the point of use.
- Avoid eager re-exports that make it mandatory elsewhere.
- Add an isolated import or installation check when the dependency graph is fragile.
- Document downloads, cache behavior, and known profile conflicts.

Before opening a PR, follow the verification commands in the repository `AGENTS.md`.

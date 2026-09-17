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

## Add an optional backend

- Put the dependency in a meaningful optional profile.
- Raise an actionable import error at the point of use.
- Avoid eager re-exports that make it mandatory elsewhere.
- Add an isolated import or installation check when the dependency graph is fragile.
- Document downloads, cache behavior, and known profile conflicts.

Before opening a PR, follow the verification commands in the repository `AGENTS.md`.

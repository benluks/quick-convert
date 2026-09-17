# API reference and docstring policy

Generated API reference is a planned second-stage documentation improvement. It should describe intentionally public interfaces rather than every importable implementation detail.

## Initial public scope

- `quick_convert.data`
- `quick_convert.data.resources`
- `quick_convert.systems`
- `quick_convert.inference`
- stable base classes and result types under `quick_convert.components`

Vendored modules under `quick_convert.external`, compatibility shims, and experiment-specific implementations should be excluded unless deliberately promoted.

## Docstring requirements

Public classes and functions should document:

- purpose and architectural role;
- parameters and accepted forms;
- tensor shapes, dtypes, devices, and valid lengths where applicable;
- return type and meaningful fields;
- mutations and side effects;
- optional dependencies, downloads, and cache behavior;
- exceptions that represent normal caller errors;
- a short example when usage is not obvious.

Docstrings should explain contracts that cannot be recovered from the type signature. They should not repeat implementation details or promise support for accidental internals.

## Current coverage

The first enforced tranche covers exports from:

- `quick_convert.data`
- `quick_convert.data.resources`
- `quick_convert.inference`

These modules form the basic path from loading inputs to saving or loading an
inference system. Tests require every exported callable in this tranche to have
a docstring. System exports and the content- and speaker-encoder base contracts
are also documented, including their optional dependencies and download
behavior. Concrete experimental implementations remain outside the stability
promise.

## Rollout

1. Extend enforced coverage to optional system and component exports without
   making documentation builds install every backend.
2. Add docstring linting after the initial backlog is addressed.
3. Generate reference pages from the curated public set.
4. Link generated reference from the hand-written task guides rather than replacing them.

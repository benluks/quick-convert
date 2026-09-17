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

## Rollout

1. Inventory exported symbols and mark the intended public set.
2. Add consistent docstrings module by module.
3. Add docstring linting only after the initial backlog is addressed.
4. Generate reference pages from the curated public set.
5. Link generated reference from the hand-written task guides rather than replacing them.

# Working in Quick Convert

## Start here

- Read `docs/index.md` for the user-facing documentation map.
- Read `docs/design_philosophy.md` before moving responsibilities between modules.
- Treat `quick_convert/configs/run/` as the list of supported executable composition roots.

## Architecture rules

- Components are focused reusable building blocks.
- Treat Python constructors as the source of truth for behavioral defaults;
  YAML should express composition, environment values, and intentional
  overrides rather than duplicate those defaults.
- Do not use catch-all constructor arguments to hide stale config fields.
- Represent values fixed by an underlying model as class constants and
  read-only instance properties, not configurable constructor arguments.
- Systems expose complete inference behavior and must not depend on pipelines or training frameworks.
- Training modules add losses, logging, and optimization around systems.
- Pipelines orchestrate datasets, execution, persistence, and other workflow concerns.
- Use the top-level Hydra key `system` for the inference object. Do not reintroduce `architecture.system` in new configs.
- Keep sample-specific annotations and precomputed features in named resources rather than adding fields to `AudioSample`.
- Reach vendored code under `quick_convert.external` through first-party adapters.

## Public behavior

- Keep optional dependencies lazy. Importing an unrelated public module must not require an optional backend.
- When a configured target adds or changes an optional backend, update
  `quick_convert/configs/dependencies.yaml`; keep package requirements and
  versions in `pyproject.toml`.
- Do not accept public arguments that are ignored.
- Preserve caller-owned tensors and objects unless mutation is explicitly documented.
- Report valid sequence lengths alongside padded tensors.
- Keep inference artifacts independent of optimizer, scheduler, logger, callback, and other training-only state.
- Ensure runtime configs and required non-Python assets are included in the installed wheel.

## Changes

- Prefer extending an existing concept over introducing a synonym or parallel abstraction.
- Add or update a canonical example when public behavior changes.
- Update the relevant task guide, configuration reference, or extension guide in the same change.
- Do not present experimental or vendored internals as stable public API.

## Verification

Run before committing:

```bash
uv run ruff format .
uv run ruff check --select B,E4,E7,E9,F,I,UP .
uv run pytest
```

For packaging, CLI, or config changes, also build and inspect an installed wheel:

```bash
uv build --wheel
python -m venv /tmp/quick-convert-wheel-smoke
/tmp/quick-convert-wheel-smoke/bin/pip install --no-deps dist/*.whl
cd /tmp
/tmp/quick-convert-wheel-smoke/bin/quick-convert --help
```

Do not declare a full integration verified unless the required model downloads, data, and optional dependency profile were actually exercised.

# AGENTS.md

## Purpose

Quick Convert is an active speech-privacy research codebase and a Python library. Its structure contains current, transitional, experimental, and legacy code. Preserve useful research behavior while improving clarity and reliability incrementally.

This document is a living set of guardrails. Update it only when a convention has been established through the code or discussion; do not encode speculative architecture as policy.

## Working approach

- Inspect and understand a path before restructuring it.
- Prefer small, reviewable changes over broad cleanup.
- Separate confirmed defects from architectural suggestions.
- Ask before making changes that choose a new abstraction, move modules, remove code, or alter experimental behavior.
- Do not infer that code is obsolete solely from its age, name, location, or lack of documentation.
- Classify uncertain code as current, transitional, legacy, experimental, or unknown before deleting or relocating it.
- Preserve unrelated changes and avoid opportunistic rewrites.

## Provisional reference workflows

The following run configurations are the current starting points for understanding the codebase. They are reference paths, not declarations that every detail is correct:

- `configs/run/train_vq_asr_librispeech.yaml`
- `configs/run/train_sslr_w2vbert_cmdiff_rvq.yaml`
- `configs/run/build_manifest_libri.yaml`
- `configs/run/precompute_content_w2vbert_librispeech.yaml`

Trace Hydra composition from these run configurations through pipelines, trainers or systems, datasets, and components before changing their dependencies.

## Architectural direction

The intended conceptual layers are:

- **Components:** reusable model and signal-processing building blocks.
- **Systems:** task-level or model-level assemblies of components.
- **Pipelines:** workflows that coordinate data, systems, execution, and outputs.

The repository has not completed this separation. Treat existing placement as evidence of current practice, not necessarily the final design, and do not perform large moves merely to match the conceptual model.

## Library-first behavior

- Important behavior should be usable through ordinary Python functions and classes.
- Hydra configuration should compose library functionality rather than contain the only implementation of it.
- Keep core logic independent of CLI parsing, filesystem layout, and Hydra where practical.
- Prefer thin pipeline and CLI wrappers around reusable functions.
- Maintain explicit public APIs through focused package exports; do not re-export every implementation at the top level.
- Avoid making base-package imports require unrelated optional dependencies.
- Treat tensor shapes, lengths, masks, sample rates, frame rates, and resource meanings as part of an interface contract.

## Configuration and ownership

- Put a setting with the component that owns and applies the behavior.
- Let architecture and run configs override component defaults when an experiment requires it.
- Trainers and pipelines should not duplicate component-owned loss weighting or model behavior.
- When modifying Hydra configuration, verify the full composition path and the target constructor arguments.

## Data and resources

- Preserve the distinction between metadata, unloaded resource references, loaded sample resources, and collated batch resources.
- Functions operating on manifests should be usable independently of a particular CSV path when feasible.
- Dataset splits involving speakers must prevent speaker overlap and should be deterministic when given a seed.
- Do not silently discard existing precomputed artifacts or manifest entries during resume operations.

## External code

Code under `quick_convert/external/` originates from external projects or integrations. Preserve provenance and isolate project-specific adaptations. Do not casually format, reorganize, or rewrite vendored code together with first-party cleanup.

## Validation

- Add focused tests around confirmed contracts and regressions.
- Test public Python APIs directly; do not rely only on end-to-end Hydra runs.
- Keep smoke tests, unit tests, and expensive model/integration tests distinguishable.
- Do not require model downloads, private datasets, GPUs, or external services in the default fast test suite.
- Run applicable formatting, linting, tests, and type checks after changes once those checks are established.

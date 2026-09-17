# Core concepts

Quick Convert uses a small set of roles. These are responsibility boundaries, not a requirement that every workflow instantiate every layer.

| Concept | Responsibility | Typical location |
|---|---|---|
| Dataset | Discover samples and load requested values | `quick_convert.data` |
| Resource | Named annotation or sidecar feature | `quick_convert.data.resources` |
| Component | Reusable model or signal-processing piece | `quick_convert.components` |
| System | Complete task-level inference behavior | `quick_convert.systems` |
| Training module | Losses, logging, and optimization around a system | `quick_convert.training` |
| Pipeline | Workflow orchestration and persistence | `quick_convert.pipelines` |
| Run config | Hydra composition root for an executable workflow | `quick_convert/configs/run` |
| Inference artifact | Versioned system configuration and weights | `quick_convert.inference` |

## The usual flow

1. A dataset discovers audio and attaches resource references.
2. A pipeline asks the dataset to load the values required by the workflow.
3. A system performs the task, possibly by composing several components.
4. During training, a training module adds objectives and optimization without changing the system's inference API.
5. The trained system can be exported without training-only state.

## Important distinctions

- A system is usable from Python without its pipeline.
- A pipeline may use a component directly when no task-level system is needed, as in feature precomputation.
- Hydra constructs objects but does not define their architectural role.
- Padded tensors should travel with their valid lengths.
- Resources keep experimental annotations out of the core sample schema.

For the deeper rationale, see [Design philosophy](../design_philosophy.md).

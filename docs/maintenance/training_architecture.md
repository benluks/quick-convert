# Training architecture inventory

This document maps the current training path and proposes a boundary between
inference-ready systems and training infrastructure. It is an architectural
checkpoint, not approval for a broad refactor.

## Current execution path

1. A run config selects `pipeline: training`, a trainer, an architecture, and
   datasets.
2. Hydra constructs the task model beneath `architecture.system`; the selected
   trainer module wraps that system.
3. `TrainingPipeline` asks the selected trainer to build itself, writes the
   fully resolved run config, and invokes training.
4. `LightningTrainer` performs dataset-dependent module setup, constructs the
   Lightning trainer, optionally compiles submodules, and calls `fit()`.
5. `BaseTrainingModule` supplies Lightning steps, optimization, logging, and
   checkpoint filtering.

The orchestration boundary is reasonable, but the object called `module` is
currently both the task model and its Lightning training adapter.

## Current ownership

| Concern | Current owner | Architectural observation |
| --- | --- | --- |
| Dataset and output orchestration | `TrainingPipeline` | Pipeline responsibility. |
| Dataloaders, DDP, precision, compilation, and `fit()` | `LightningTrainer` | Training runtime responsibility, not a task system. |
| Optimizer and scheduler construction | `Optimization` and `BaseTrainingModule` | Training-only state. |
| Model architecture | Plain systems under `architecture.system` | Supported VQ-ASR and SSL-reconstruction configurations construct systems independently of Lightning. |
| Task inference | Plain systems | VQ-ASR and SSL reconstruction expose typed, Lightning-independent results. |
| Losses and training metrics | Concrete Lightning modules and some components | Training behavior is interleaved with model execution. |
| Media and gradient logging | Lightning modules and mixins | Training-only behavior. |
| Frozen online resource encoders | Concrete Lightning modules | Needed by some inference paths, but deliberately omitted from some checkpoints. |
| Checkpoint reconstruction | `BaseTrainingModule.from_run()` | Re-instantiates `cfg.pipeline.trainer.module`, including training-only arguments. |

`TokenizerTrainer` is a separate case: it adapts a dataset to a tokenizer's
training API. It belongs with training infrastructure, but its trained tokenizer
artifact should remain usable without that adapter.

## Why inference loading is fragile

`BaseTrainingModule.from_run()` reads `config.yaml`, instantiates
`cfg.pipeline.trainer.module`, then loads the Lightning `state_dict`. This means
that inference reconstruction depends on the historical constructor for the
entire training module. Adding or removing a loss weight, optimizer object,
logger-related option, or dataset-dependent field can prevent an otherwise
compatible model from loading. `strict=False` can hide state-key differences,
but it does not establish a stable inference contract.

Checkpoint filtering adds another layer: excluded online encoders must be
reconstructed from the current configuration before strict loading. That is a
reasonable storage policy, but it should be expressed in an inference artifact
manifest rather than inferred from a Lightning module's training constructor.

## Proposed boundary

### Systems

Plain `torch.nn.Module` implementations under `quick_convert.systems` should
own the complete task capability:

- a VQ-ASR system owns feature resolution, layer fusion, quantization,
  contextualization, and the CTC prediction head;
- an SSL-reconstruction system owns feature resolution, optional RVQ encoding,
  decoding, and waveform generation; and
- each system exposes a typed, training-independent forward/inference result.

Frozen online encoders may still be system dependencies when they are required
to run from audio. Whether their weights are bundled or reconstructed is an
artifact policy, not a reason to make the system a Lightning module.

### Lightning adapters

Lightning modules should wrap a system and own only training behavior:

- target extraction and objective computation;
- loss weighting;
- optimizer and scheduler configuration;
- scalar, media, ASR, and gradient logging;
- training/validation hooks; and
- resumable Lightning checkpoint behavior.

The wrappers may remain temporarily under `pipelines.training.modules` during
migration, but their long-term home should be a framework-specific training
package such as `quick_convert.training.lightning`. They are neither pipelines
nor task systems.

### Pipelines and trainers

`TrainingPipeline` should continue to coordinate datasets, output paths,
configuration capture, and execution. A Lightning runner and a SentencePiece
runner are training backends/adapters. Naming them `systems` would blur the
existing meaning of a system as a task-level capability.

## Configuration boundary

The architecture config should construct a system independently:

```yaml
architecture:
  system:
    _target_: quick_convert.systems.asr.VQASRSystem
    quantizer: ...
    ctc_head: ...
```

The trainer config should then wrap it:

```yaml
trainer:
  module:
    _target_: quick_convert.pipelines.training.modules.vq_asr.VQASRTrainingModule
    system: ${architecture.system}
    optimization: ...
    ctc_loss_weight: 1.0
```

This allows the same `architecture.system` node to be instantiated for
inference without resolving optimizer, loss-weight, logger, or trainer config.

## Artifact boundary

Two artifacts serve different purposes and should not be conflated:

| Artifact | Contents | Intended use |
| --- | --- | --- |
| Training checkpoint | System weights, optimizer/scheduler state, epochs, steps, and Lightning callback state | Exact training resumption. |
| Inference artifact | System config, system weights, format/version metadata, and declarations for excluded external weights | Portable inference and library use. |

An inference loader should instantiate `architecture.system` and load only the
system state. Loading should not require `cfg.pipeline.trainer.module`.
Initially, an export helper can extract `system.*` keys from existing Lightning
checkpoints; future checkpoints can save the system state explicitly.

## Recommended migration sequence

1. ~~Define a typed VQ-ASR inference output.~~ SSL reconstruction still needs
   its equivalent.
2. ~~Extract `VQASRSystem` and verify that it returns CTC logits as well as
   useful intermediate representations.~~
3. ~~Make `VQASRTrainingModule` wrap the system, retain legacy construction,
   and convert historical checkpoint keys explicitly.~~
4. ~~Expose `architecture.system` in the supported VQ-ASR Hydra config.~~
5. ~~Extract `SSLReconstructionSystem`, move file/tensor inference conveniences
   out of the Lightning module, and expose it as `architecture.system` in each
   supported SSL-reconstruction configuration.~~
6. Introduce a versioned inference artifact loader/exporter.
7. Only after those contracts stabilize, move Lightning-specific modules and
   runners out of `pipelines` into a dedicated training package.

## Decisions to workshop

- Should VQ-ASR inference return only logits, or a typed result containing
  logits, lengths, quantizer output, and contextual features?
- Should objective objects live wholly in Lightning adapters, or may supervised
  heads retain `compute_loss()` convenience methods while systems call only
  their prediction path?
- Should an inference artifact bundle frozen online encoder weights, reference
  their upstream model identifiers, or support both policies explicitly?
- How much compatibility is required for current Lightning checkpoints whose
  keys are not prefixed by `system.`?
- Is `quick_convert.training` the desired long-term home for framework adapters,
  or should this remain deliberately narrower until another training backend
  exists?

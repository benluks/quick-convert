# Loading systems for inference

Use one loader for a training run directory, a checkpoint file, or an exported
inference artifact:

```python
from quick_convert.inference import load_inference_system

system = load_inference_system("outputs/my_experiment/run", map_location="cpu")
system = load_inference_system("outputs/my_experiment/run/checkpoints/best.ckpt")
system = load_inference_system("exports/my_system")
```

Each form constructs only the configured inference system and loads its
weights. It returns a frozen `torch.nn.Module` in evaluation mode, on CPU by
default. Training modules, optimizers, schedulers, callbacks and loggers are
not constructed. Loading does not write an intermediate artifact or modify the
source run.

## Checkpoint and configuration selection

For a run directory, the default checkpoint is `checkpoints/last.ckpt`:

```python
from quick_convert.inference import load_system_from_run

system = load_system_from_run(
    "outputs/my_experiment/run",
    checkpoint="checkpoints/best.ckpt",
    map_location="cpu",
)
```

New checkpoints written through `TrainingPipeline` carry a resolved inference
composition under `quick_convert_inference`. These checkpoints can be moved
away from the run directory and loaded on their own. Only the system
composition is embedded; training configuration is not part of this metadata.
Paths or external assets referenced by that composition still need to be
available.

Older runs fall back to their `config.yaml`. When given an older checkpoint
file, the loader looks beside it, then in its parent run directory if the
checkpoint is inside a directory named `checkpoints`. It does not search other
ancestors. An explicit config overrides both embedded metadata and discovery:

```python
system = load_inference_system(
    "checkpoints/model.ckpt",
    config="/absolute/path/to/config.yaml",
)
```

Some older checkpoints pickle custom training objects in their hyperparameters.
Deserializing those files still needs the classes that originally saved them.
Export once in that environment to produce an artifact that loads independently.
New standard training checkpoints omit instantiated optimization objects from
hyperparameters.

For checkpoint-file sources, relative config paths start beside the checkpoint.
For run-directory sources, relative config and checkpoint paths start at the
run directory. Existing `architecture.system` configs remain readable for
compatibility; new runs should use top-level `system`.

An artifact already selects its configuration and weights. Supplying `config`
or `checkpoint` with an artifact raises an error, as does supplying `checkpoint`
when the source is already a checkpoint file.

## Exporting an artifact

The existing export command uses the same system extraction as direct loading:

```bash
quick-convert export outputs/my_experiment/run exports/my_system
quick-convert export outputs/my_experiment/run exports/my_system \
  --checkpoint checkpoints/best.ckpt
```

Export prefers embedded composition when present and otherwise reads
`config.yaml`. Use `--config alternate.yaml` to explicitly override it.
The artifact contains `artifact.yaml` and `weights.pt`, with no optimizer,
logger, scheduler or training-only parameters. Artifact format version 1 stays
compatible with existing exports.

Loading is strict by default. Missing or unexpected weights raise an error.
Explicitly excluded prefixes are filled from the newly constructed system,
matching existing artifact behavior. Such omissions require reproducible
constructor state, such as a fixed pretrained backend. A new self-describing
checkpoint cannot omit a trainable online encoder; for VQ-ASR, enable
`trainer.module.save_online_encoders=true` when fine-tuning those encoders.
`strict=False` deliberately allows missing or unexpected state keys, but still
checks incompatible tensor shapes.

## System classes and extensions

Existing ASR, reconstruction, anonymization and SER system bases inherit
`InferenceSystem`. Their class methods check the restored system's type:

```python
from quick_convert.systems.asr import VQASRSystem

system = VQASRSystem.from_run("outputs/my_experiment/run")
system = VQASRSystem.from_artifact("exports/my_system")
```

New task models can subclass `quick_convert.systems.base.InferenceSystem` to
inherit these methods. The general `load_inference_system` function also works
with any Hydra-configured `torch.nn.Module`, without this base class.

Use top-level `system` in the run config and register the model as `self.system`
in its training wrapper. This produces `system.*` checkpoint keys. The shared
loader also accepts a plain system state dictionary and removes compiled-module
key wrappers before restoration. It does not infer systems from arbitrary
training-module configurations or migrate older model-specific parameter names.

Programmatic `BaseTrainingModule` users can embed composition directly before
checkpointing:

```python
module.set_inference_config({"system": resolved_system_config})
```

The standard pipeline supplies this automatically. Training-only compositions
without `system` keep producing ordinary training checkpoints.

When Hydra uses a loader as a `_target_`, dependency planning inspects the saved
system composition without instantiating it. Required optional backends follow
that system, rather than its former trainer. Exported artifacts can be inspected
using their manifest alone; run and checkpoint inspection reads the checkpoint
on CPU.

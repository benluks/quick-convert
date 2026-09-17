# Using Quick Convert from Python

Pipelines and Hydra entry points are conveniences. Datasets, resources, systems, components, and inference artifacts are ordinary Python APIs.

## Discover and batch audio

```python
from quick_convert.data import BaseDataset

dataset = BaseDataset(
    root="/data/librispeech/Librispeech",
    splits=["test-clean"],
    file_format="flac",
    utt_id_template="{path.stem}",
    load=["audio"],
    target_sr=16_000,
)

loader = dataset.make_dataloader(batch_size=8, num_workers=4)
batch = next(iter(loader))
print(batch.waveforms.shape, batch.lengths)
```

Exactly one of `root`, `paths`, or `rows` identifies the dataset source. Audio is loaded only when `load` contains `audio` or is set to `all`.

## Attach a sidecar resource

```python
from quick_convert.data import BaseDataset
from quick_convert.data.resources import PathResourceProvider

content = PathResourceProvider(
    name="content",
    kind="torch_tensor",
    path_template="outputs/precomputed/librispeech/content/{sample.split}/{sample.utt_id}.pt",
)

dataset = BaseDataset(
    root="/data/librispeech/Librispeech",
    splits=["test-clean"],
    file_format="flac",
    utt_id_template="{path.stem}",
    resource_providers=[content],
    load=["audio", "content"],
)
```

Providers describe where a value comes from. The dataset decides whether the resulting reference is loaded.

## Compose a configured component

```python
from hydra.utils import instantiate
from quick_convert.utils.config import compose_component

config = compose_component("components/ssl", "w2vbert", {"device": "cpu"})
encoder = instantiate(config)
```

This uses the config tree installed inside the package. Model-specific optional dependencies and downloads still apply.

## Load an inference artifact

```python
from quick_convert.inference import load_inference_artifact

system = load_inference_artifact("models/vq-asr", map_location="cpu")
result = system(batch)
```

The concrete result type depends on the system. For example, `VQASRSystem` returns logits, valid lengths, codes, quantized values, and probeable intermediate representations.

## Save an instantiated system

```python
from quick_convert.inference import save_inference_artifact

save_inference_artifact(
    system,
    system_config,
    "models/my-system",
)
```

`system_config` must be a resolved Hydra-compatible mapping containing `_target_`. Use `excluded_state_prefixes` for intentionally external online encoders or other reproducible state that should not be bundled.

## Export a training run

```python
from quick_convert.inference import export_inference_artifact

export_inference_artifact(
    "outputs/my-run",
    "models/my-system",
    checkpoint="checkpoints/last.ckpt",
)
```

The run directory must contain its resolved `config.yaml` and checkpoint. The resulting artifact excludes optimizer, scheduler, logger, and callback state.

## Stability boundary

Public first-party interfaces under `quick_convert.data`, `quick_convert.systems`, and `quick_convert.inference` are the preferred library surface. Component modules are reusable but more architecture-specific. `quick_convert.external` is vendored implementation detail.

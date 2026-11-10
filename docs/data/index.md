# Data and resources

Quick Convert separates sample discovery, resource location, loading, and collation so the same audio dataset can be reused with different annotations and precomputed features.

`AudioSample` identifies an utterance. `AudioBatch` contains padded waveforms, valid lengths, sample rate, IDs, and requested resources. `BaseDataset` discovers audio; `ManifestDataset` reads explicit CSV rows for stable splits and portable metadata.

Resources are named values such as `transcript`, `token_ids`, `content`, or `speaker_embedding`. Providers map samples to sidecar files, templates, or CSV annotations. Dataset `load` settings decide which resources are materialized.

Hydra configs are installed under `quick_convert/configs/dataset/` and `quick_convert/configs/resource_providers/`.

For constructor, discovery, and sample options, see the [BaseDataset reference](base_dataset.md).

## Manifests

Build and split the LibriSpeech manifest within each speaker:

    uv run quick-convert build_flat_manifest_libri data_root=/data
    uv sync --extra manifests
    uv run python -m quick_convert.cli.split_manifest \
        --input outputs/librispeech/manifest.csv \
        --train-output outputs/librispeech/speaker_split/train.csv \
        --valid-output outputs/librispeech/speaker_split/val.csv

The default `within-group` strategy places rows from each speaker into both subsets.

For speaker-disjoint CLAC manifests, preserve each elicitation name in `split` and assign every
recording from a speaker to the same partition:

    export QUICK_CONVERT_CLAC_ROOT=/path/to/CLAC-Dataset
    uv run quick-convert build_flat_manifest_clac
    uv run python -m quick_convert.cli.split_manifest \
        --input outputs/clac/manifest.csv \
        --train-output outputs/clac/speaker_split/train.csv \
        --valid-output outputs/clac/speaker_split/val.csv \
        --strategy group-disjoint \
        --group-col spkid \
        --valid-fraction 0.1 \
        --seed 115

The CLAC manifest uses split-qualified utterance IDs such as `picnic/1234`, retains the
elicitation task in the `split` column, and stores `1234` as `spkid`.

Datasets and providers are ordinary Python objects, so pipelines are optional orchestration rather than a requirement.

Reusable transformations can be fitted and persisted with the [fitting workflow](../guides/fitting.md).

## Audio resources

Use `kind="audio"` for any additional recording: content speech, references,
noise, or other sidecar audio. Resource loading and collation handle audio;
consumers decide how to use it. No pairing, cropping, mixing, or augmentation
policy is imposed by the resource kind.

```python
from quick_convert.data import ManifestDataset

# CSV columns: utt_id,path,noise_path
# The noise path may be reused across multiple target utterances.
dataset = ManifestDataset(
    "manifest.csv",
    resources={
        "noise": {"column": "noise_path", "kind": "audio", "sample_rate": 16_000},
    },
    load=["audio", "noise"],
    target_sr=16_000,
)
batch = next(iter(dataset.make_dataloader(batch_size=8)))
noise = batch.resources["noise"]  # AudioBatch; mixing belongs to the consumer.
```

`PathResourceProvider(name, path_template, kind="audio", sample_rate=16000)`
supports template-based paths through the same loading machinery. `ResourceRef`
holds the path and optional target `sample_rate`; leaving it `None` preserves
the file rate. An unloaded reference has `value=None`. Select the resource by
name in dataset `load`, or use `load="all"`. Direct calls to `load_resource(ref)`
produce an updated reference whose value is an `AudioSample`.

Audio resources are mixed to mono. A loaded sample records its actual sample
rate and waveform. Collation pads recordings into an `AudioBatch` with exact
lengths, source paths, and the corresponding target IDs and splits. Different
native rates remain explicit in `sample_rates`; consumers needing a common
rate should configure it on the resource. Additional audio can be loaded even
when the target waveform is not loaded. Unloaded resources cannot be collated
as audio; select them in `load` before constructing batches.

The usual `batch[index]` access retains loaded audio resources without keeping
separate per-example waveform storage. Standard loading uses CPU tensors;
Lightning handles transfer with the rest of the batch.

### Cached content representations

Content precomputation writes versioned tensor payloads with representation
metadata, including the frame rate when known. Load them with the existing
`torch_tensor` resource kind. Batches retain the metadata and reject mixtures
of incompatible representations. Legacy bare tensors still load with unknown
representation metadata. To regenerate an existing content cache in the new
format, select a fresh precompute output directory. Code that reads the new
files directly with `torch.load` can access the tensor under the `values` key.

## Group-disjoint cross-validation

Use one group ID per item in dataset order, such as a speaker ID or a source
recording ID. Assign folds once and reuse the mapping for each run. The
assignment balances item counts and keeps all items from each group together.

```python
from torch.utils.data import DataLoader, Subset

from quick_convert.data.cross_validation import assign_folds, fold_indices

group_ids = [row.resources.spkid.value for row in dataset.rows]
assignments = assign_folds(group_ids, n_splits=5, seed=115)
train_indices, held_out_indices = fold_indices(group_ids, assignments, fold=0)

train = DataLoader(Subset(dataset, train_indices), collate_fn=dataset.collate_fn)
held_out = DataLoader(Subset(dataset, held_out_indices), collate_fn=dataset.collate_fn)
```

This example uses speaker IDs already attached to metadata rows. Fold selection
uses row indices and does not load audio or rewrite the manifest.

## Build dataset rows from tables and RTTM

Supply the result of an index to `BaseDataset(rows=...)` to construct a
dataset from recording metadata and RTTM annotations. The
[example Hydra config](examples/rttm_dataset.yaml)
reads selected CSV columns, splits compound recording IDs into path fields,
links equivalent identities, joins the tables,
and selects a configured speaker label per recording. The example supplies
an RTTM reader to the segment builder. Other annotation formats can supply
their own reader returning start and end times in seconds.

The example expects `takes.csv` with `take,reader,reader_track` and
`readers.csv` with `reader,aliases`. For a take such as `studio-003`, the
index derives `studio` and `003` from the ID to locate `audio/studio/003.wav`.
The annotations are named `annotations/<take>.rttm`. Aliases are
semicolon-separated reader IDs. A blank alias field adds no links.

```python
from hydra.utils import instantiate
from omegaconf import OmegaConf

config = OmegaConf.load("docs/data/examples/rttm_dataset.yaml")
config.data_root = "/path/to/recordings"
dataset = instantiate(config.dataset)
clip = dataset[0]
```

Each selected turn becomes a row with its source path, time bounds, and
configured resources. The index reads annotation metadata at construction.
Audio is loaded when the row is accessed. This example uses individual RTTM
turns as annotated, including any overlap present in those turns.


### Select a fold from config

Replace `dataset.rows` in the example with the following mapping to select
one partition from the complete `segment_index`:

```yaml
rows:
  _target_: quick_convert.data.cross_validation.select_fold_rows
  rows: ${segment_index}
  group_resource: spkid
  n_splits: 2
  fold: 0
  partition: held_out
```

Use `partition: train` for the complement. Both configs must use the same
complete index and fold settings. Folds are balanced by item count, and
linked speakers stay together even if their recordings have different IDs.

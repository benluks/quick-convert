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

# Data and resources

Quick Convert separates sample discovery, resource location, loading, and collation so the same audio dataset can be reused with different annotations and precomputed features.

`AudioSample` identifies an utterance. `AudioBatch` contains padded waveforms, valid lengths, sample rate, IDs, and requested resources. `BaseDataset` discovers audio; `ManifestDataset` reads explicit CSV rows for stable splits and portable metadata.

Resources are named values such as `transcript`, `token_ids`, `content`, or `speaker_embedding`. Providers map samples to sidecar files, templates, or CSV annotations. Dataset `load` settings decide which resources are materialized.

Hydra configs are installed under `quick_convert/configs/dataset/` and `quick_convert/configs/resource_providers/`.

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

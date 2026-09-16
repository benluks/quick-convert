# Data and resources

Quick Convert separates sample discovery, resource location, loading, and collation so the same audio dataset can be reused with different annotations and precomputed features.

`AudioSample` identifies an utterance. `AudioBatch` contains padded waveforms, valid lengths, sample rate, IDs, and requested resources. `BaseDataset` discovers audio; `ManifestDataset` reads explicit CSV rows for stable splits and portable metadata.

Resources are named values such as `transcript`, `token_ids`, `content`, or `speaker_embedding`. Providers map samples to sidecar files, templates, or CSV annotations. Dataset `load` settings decide which resources are materialized.

Hydra configs are installed under `quick_convert/configs/dataset/` and `quick_convert/configs/resource_providers/`.

## Manifests

    uv run quick-convert build_flat_manifest_libri data_root=/data
    uv sync --extra manifests
    mkdir -p outputs/librispeech/speaker_split
    uv run python -m quick_convert.cli.split_manifest \
        --input outputs/librispeech/manifest.csv \
        --train-output outputs/librispeech/speaker_split/train.csv \
        --valid-output outputs/librispeech/speaker_split/val.csv

The splitter places rows from each speaker into both subsets by default; choose another policy for speaker-disjoint evaluation. Datasets and providers are ordinary Python objects, so pipelines are optional orchestration rather than a requirement.

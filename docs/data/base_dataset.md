# BaseDataset

`BaseDataset` provides the common path discovery, resource attachment, loading,
and batching behavior used by Quick Convert datasets. It can be constructed
from exactly one of three sources:

- a root directory to scan recursively;
- an explicit iterable of audio paths;
- an iterable of existing `MetadataSample` rows.

The dataset initially represents each item as lightweight metadata. Audio and
other resources are materialized only when requested through `load`.

## Samples

The fields intrinsic to a sample are deliberately small:

```python
MetadataSample(
    utt_id="84-121123-0001",
    path=Path("84/121123/84-121123-0001.flac"),
    split="train-clean-100",
    resources=ResourceCollection(...),
)
```

`AudioSample` adds only `waveform` and `sample_rate`. Corpus annotations and
experiment-specific values—including speaker identity, transcripts, token IDs,
and precomputed features—belong in the named resource collection rather than
as dedicated sample fields.

## Construction

When scanning a directory, an utterance-ID rule must be supplied:

```python
dataset = BaseDataset(
    root="data/LibriSpeech",
    splits=["train-clean-100"],
    file_format="flac",
    utt_id_template="{path.stem}",
)
```

Already-selected files can be passed directly:

```python
dataset = BaseDataset(
    paths=selected_paths,
    get_utt_id_fn=lambda path: path.stem,
)
```

Existing metadata rows are useful when selection or metadata parsing happens
outside filesystem discovery:

```python
dataset = BaseDataset(rows=manifest_rows)
```

Passing none or more than one of `root`, `paths`, and `rows` raises a
`ValueError`.

## Resources and loading

Resource providers attach arbitrary named `ResourceRef` objects to each sample.
The `load` argument controls which values are materialized:

- `False` or `None`: retain references without loading them;
- `True` or `"all"`: load audio and every configured resource;
- a list of names: load only those resources (use `"audio"` for audio).

```python
dataset = BaseDataset(
    root="data/speech",
    file_format="wav",
    utt_id_template="{path.stem}",
    resource_providers=[speaker_provider, transcript_provider],
    load=["audio", "transcript"],
)
```

This keeps corpus discovery independent from the annotations and features used
by a particular experiment.

## Batching

`make_dataloader()` uses the dataset's collation function to produce an
`AudioBatch`. Loaded waveforms are padded along time and retain their original
lengths. Named resources are collated according to their resource kind.

The base class is intended to remain corpus-agnostic. Corpus-specific code may
construct rows, configure providers, or subclass the dataset when discovery
rules differ, but should not add experiment-specific metadata to the shared
sample contract.

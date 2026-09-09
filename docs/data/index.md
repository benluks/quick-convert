# Data

`quick_convert.data` provides reusable dataset, resource, loading, and batching abstractions for speech experiments.

The module is designed around a simple idea: an audio sample consists of a small amount of core metadata—such as its path, utterance ID, and split—plus an arbitrary collection of named **resources**. Resources may be immediately available values, references to files that should be loaded on demand, or features computed online.

This allows datasets to remain generic. A dataset does not need dedicated fields for transcripts, speaker IDs, SSL features, token sequences, embeddings, or experiment-specific annotations. Instead, these values are attached through resource providers and accessed uniformly through the sample's resource collection.

A typical data flow is:

```text
audio files / manifest
        ↓
    BaseDataset
        ↓
 MetadataSample
        +
 resource providers
        ↓
   ResourceRef
        ↓
 optional loading
        ↓
    AudioSample
        ↓
    collation
        ↓
     AudioBatch
```

## Core concepts

The data module separates four concerns:

1. **Datasets** determine which samples exist and provide basic sample metadata.
2. **Resource providers** determine which resources belong to each sample.
3. **Resource references and loaders** describe and optionally materialize those resources.
4. **Batch collation** combines samples and variable-length resources into model-ready batches.

This separation makes it possible to reuse the same dataset definition with different experiment-specific resources.

For example, LibriSpeech might be used once with transcripts and token IDs:

```text
LibriSpeech
├── transcript
└── token_ids
```

and elsewhere with SSL features and acoustic measurements:

```text
LibriSpeech
├── wavlm
├── pitch
└── speaker_id
```

without requiring a different dataset class.

Experiments are configured through Hydra, allowing architectures, datasets, feature extractors, trainers, and evaluation systems to be composed without rewriting the surrounding infrastructure.

# Samples and batches

## `MetadataSample`

`MetadataSample` contains the metadata required to identify an utterance:

```python
MetadataSample(
    utt_id="84-121123-0001",
    path=Path("84/121123/84-121123-0001.flac"),
    split="train-clean-100",
)
```

Its principal fields are:

```text
utt_id
    Unique utterance identifier.

path
    Path to the corresponding audio file.

split
    Optional dataset split.

resources
    ResourceCollection associated with the sample.
```

Samples are frozen dataclasses. Operations that add audio or resources therefore return updated samples rather than mutating the existing object.

## `AudioSample`

`AudioSample` extends sample metadata with optionally loaded audio:

```python
sample.waveform
sample.sample_rate
```

An unloaded `AudioSample` may therefore contain:

```python
AudioSample(
    utt_id="84-121123-0001",
    path=Path("84-121123-0001.flac"),
)
```

while a loaded sample additionally contains its waveform and sampling rate.

A sample can also be constructed directly from a path:

```python
from quick_convert.data import AudioSample

sample = AudioSample.from_path("example.wav")
```

and loaded explicitly:

```python
sample = sample.load_audio(
    target_sr=16_000,
    mono=True,
)
```

## `AudioBatch`

`AudioBatch` is the collated representation returned by dataset dataloaders.

It contains:

```text
utt_ids
paths
splits
resources
waveforms
lengths
sample_rates
```

If audio was not requested from the dataset, the waveform-related fields remain `None`.

When audio is present, variable-length waveforms are padded along the time dimension:

```python
batch.waveforms
# [B, T_max]

batch.lengths
# [B]
```

`lengths` stores the original waveform length for each sample before padding.

Individual samples can be recovered by indexing the batch:

```python
sample = batch[0]
```

and batches are iterable:

```python
for sample in batch:
    ...
```

1. training a SentencePiece tokenizer;
2. precomputing token IDs;
3. building a CSV manifest;
4. training a VQ-ASR model from that manifest.

# Datasets

## `BaseDataset`

`BaseDataset` is the main filesystem-backed dataset implementation.

It supports constructing a dataset from exactly one of:

```text
root
paths
rows
```

### From a root directory

The most common use is scanning a directory:

```python
from quick_convert.data import BaseDataset

dataset = BaseDataset(
    root="/datasets/LibriSpeech",
    splits=["train-clean-100"],
    file_format="flac",
    utt_id_template="{path.stem}",
)
```

When `splits` are supplied, each split is interpreted as a directory below `root`:

```text
/datasets/LibriSpeech/
├── train-clean-100/
├── train-clean-360/
└── dev-clean/
```

Multiple splits may be combined:

```python
dataset = BaseDataset(
    root="/datasets/LibriSpeech",
    splits=[
        "train-clean-100",
        "train-clean-360",
    ],
    file_format="flac",
    utt_id_template="{path.stem}",
)
```

If `splits=None`, the entire root is searched.

### From explicit paths

A dataset may instead be constructed from a collection of audio paths:

```python
dataset = BaseDataset(
    paths=[
        "sample1.wav",
        "sample2.wav",
    ],
    utt_id_template="{path.stem}",
)
```

### From rows

Higher-level dataset implementations can construct `MetadataSample` objects directly and pass them to `BaseDataset`:

```python
dataset = BaseDataset(
    rows=rows,
)
```

This is how `ManifestDataset` operates.

## File discovery

`file_format` restricts discovery to supported audio formats:

```python
file_format = "wav"
```

or:

```python
file_format = ["wav", "flac"]
```

Formats may include or omit the leading period.

A glob-like `pattern` further restricts discovered files:

```python
pattern = "*.flac"
```

and `exclude_patterns` can remove matching files:

```python
exclude_patterns = [
    "*noise*",
    "*/excluded/*",
]
```

## Utterance IDs

Each discovered sample requires an utterance ID.

The preferred lightweight mechanism is a template:

```python
utt_id_template = "{path.stem}"
```

For more specialized behavior, a function may be supplied:

```python
dataset = BaseDataset(
    ...,
    get_utt_id_fn=lambda path: path.stem.upper(),
)
```

At least one mechanism must be available when samples are discovered from paths.

## Sorting

Rows are sorted after discovery.

The default is:

```python
sort_key = "{row.path}"
```

A different template may be supplied when another ordering is useful.

* training;
* evaluation;
* anonymization;
* feature precomputation;
* manifest generation.

# Loading audio

Dataset construction and audio loading are deliberately separate.

By default:

```python
load = False
```

means accessing a sample does not read its waveform from disk.

To load audio:

```python
dataset = BaseDataset(
    ...,
    load=["audio"],
)
```

or:

```python
load = True
```

to load audio together with all configured resources.

Audio loading supports optional resampling:

```python
target_sr = 16_000
```

and mono conversion:

```python
convert_to_mono = True
```

For example:

```python
dataset = BaseDataset(
    root="/data/speech",
    file_format="wav",
    utt_id_template="{path.stem}",
    target_sr=16_000,
    load=["audio"],
)
```

Then:

```python
sample = dataset[0]

sample.waveform
sample.sample_rate
```

contain the loaded 16 kHz audio.

* automatic speech recognition;
* automatic speaker verification;
* speech anonymization and voice conversion.

# Resources

Resources are arbitrary values associated with a sample.

Examples include:

```text
transcript
speaker_id
token_ids
wavlm
emotion_embedding
pitch
prosody
language
```

Resources are represented at sample level by `ResourceRef`.

## `ResourceRef`

A resource reference describes a named resource:

```python
ResourceRef(
    name="wavlm",
    kind="torch_tensor",
    path=Path("features/example.pt"),
)
```

or an already-materialized value:

```python
ResourceRef(
    name="speaker_id",
    kind="text",
    value="84",
)
```

Important fields are:

```text
name
    Name used to access the resource.

kind
    Determines how the resource is loaded and collated.

path
    Optional location of a serialized resource.

value
    Optional materialized value.

max_length
    Optional fixed padding length for tensor resources.
```

A resource can therefore exist in either an unresolved state:

```python
ResourceRef(
    name="wavlm",
    kind="torch_tensor",
    path=Path("84-121123-0001.pt"),
    value=None,
)
```

or a materialized state:

```python
ResourceRef(
    name="wavlm",
    kind="torch_tensor",
    path=Path("84-121123-0001.pt"),
    value=features,
)
```

Components are reusable model and signal-processing building blocks. They are analogous to PyTorch `nn.Module` objects and may be composed recursively.

# `ResourceCollection`

Every sample owns a `ResourceCollection`.

Resources can be accessed by name:

```python
sample.resources["transcript"]
```

or using attribute syntax:

```python
sample.resources.transcript
```

A collection can be created from references:

```python
resources = ResourceCollection.from_refs(
    [
        ResourceRef(
            name="speaker_id",
            kind="text",
            value="84",
        ),
        ResourceRef(
            name="wavlm",
            kind="torch_tensor",
            path=Path("features/example.pt"),
        ),
    ]
)
```

Resource names must be unique.

Collections may also be merged:

```python
resources = resources_a.merge(resources_b)
```

By default, duplicate names raise an error. Explicit overwrite behavior can be requested where appropriate.

* [Installation](installation.md)
* [Quickstart](quickstart.md)
* [Configuration and Hydra](configuration.md)
* [Running pipelines](pipelines/index.md)

# Resource providers

A resource provider defines how a resource is derived for a sample.

When an item is requested, `BaseDataset` evaluates all configured providers:

```text
sample
  ↓
provider(sample)
  ↓
ResourceRef
```

The returned references are combined with any resources already stored on the sample.

For example:

```python
dataset = BaseDataset(
    ...,
    resource_providers=[
        speaker_provider,
        transcript_provider,
        feature_provider,
    ],
)
```

Each resulting sample may then expose:

```python
sample.resources.speaker_id
sample.resources.transcript
sample.resources.wavlm
```

## `TemplateResourceProvider`

`TemplateResourceProvider` is the simplest provider.

It evaluates a template against the current sample and stores the result directly in the resource value:

```python
from quick_convert.data.resources import TemplateResourceProvider

speaker_provider = TemplateResourceProvider(
    name="speaker_id",
    template="{path.parent.parent.name}",
    kind="text",
)
```

For:

```text
/data/LibriSpeech/84/121123/example.flac
```

this might produce:

```python
ResourceRef(
    name="speaker_id",
    kind="text",
    value="84",
)
```

This is useful for metadata that can be derived directly from the sample path or other sample fields.

Typical uses include:

```text
speaker IDs
language labels
session IDs
corpus IDs
categorical metadata
```

## `PathResourceProvider`

`PathResourceProvider` resolves a path rather than an immediate value.

For example:

```python
wavlm_provider = PathResourceProvider(
    name="wavlm",
    path_template="/features/wavlm/{sample.utt_id}.pt",
    kind="torch_tensor",
)
```

For each sample it produces approximately:

```python
ResourceRef(
    name="wavlm",
    kind="torch_tensor",
    path=Path("/features/wavlm/84-121123-0001.pt"),
    value=None,
)
```

The feature is then loaded only if the dataset's `load` policy requests the `wavlm` resource.

By default, resolved paths must exist. This can be disabled with:

```python
must_exist = False
```

`max_length` may also be supplied for tensor resources when a fixed padded shape is required.

## `CSVTranscriptProvider`

`CSVTranscriptProvider` handles transcripts stored in shared CSV-like files.

Unlike `PathResourceProvider`, where each sample generally references its own feature file, a CSV annotation provider loads an annotation file into an internal lookup table and retrieves the appropriate entry for each utterance.

It supports:

```text
configurable utterance key
configurable key and text columns
custom delimiters
different encodings
joining multiple text columns
cached annotation files
```

For example:

```python
transcript_provider = CSVTranscriptProvider(
    name="transcript",
    path_template="{path.parent}/transcripts.csv",
    utterance_key="path.stem",
    key_column=0,
    text_column=1,
)
```

Files are cached after their first access, so repeated samples referring to the same transcript file do not repeatedly parse it.

This provider is useful for resources where one external file contains annotations for many utterances.

# Selective resource loading

Resource discovery and resource loading are separate operations.

A `PathResourceProvider` may attach:

```python
ResourceRef(
    name="wavlm",
    path=...,
    value=None,
)
```

without actually reading the tensor.

The dataset's `load` argument determines what is materialized.

For example:

```python
dataset = BaseDataset(
    ...,
    resource_providers=[
        transcript_provider,
        wavlm_provider,
    ],
    load=["audio", "wavlm"],
)
```

will load:

```text
audio
wavlm
```

while a transcript already supplied as an in-memory value does not require additional loading.

The special values:

```python
load = True
```

and:

```python
load = "all"
```

request audio and all configured resources.

The loading decision follows two rules:

1. A resource that already has a value does not need to be loaded.
2. An unresolved resource is loaded only if its name appears in the dataset's load set.

This makes it inexpensive to attach many possible resources to a dataset while loading only those required by a particular experiment.

---

# Resource loaders

Serialized resources are materialized through `load_resource()`.

Loading behavior is determined by resource `kind`.

For example, a tensor resource:

```python
ResourceRef(
    name="wavlm",
    kind="torch_tensor",
    path=Path("wavlm.pt"),
)
```

is loaded using the registered torch loader.

The loader returns an updated `ResourceRef`:

```python
ResourceRef(
    name="wavlm",
    kind="torch_tensor",
    path=Path("wavlm.pt"),
    value=tensor,
)
```

The loader registry makes serialization behavior extensible: support for additional resource kinds can be added by registering the corresponding loading function.

---

# Resource collation

At sample level, resources remain `ResourceRef` objects.

At batch level, resources are collated into model-ready structures:

```python
batch.resources["transcript"]
batch.resources["wavlm"]
batch.resources["token_ids"]
```

Collation depends on resource kind.

## Text

Text values are collated into a Python list:

```python
batch.resources["transcript"]
# [
#     "THE FIRST TRANSCRIPT",
#     "THE SECOND TRANSCRIPT",
#     ...
# ]
```

## Tensor resources

Variable-length tensor resources are padded along their first dimension.

A tensor resource is normalized so that its first dimension represents time.

Supported input shapes include:

```text
[D]
[T, D]
[1, T, D]
[T, L, D]
[1, T, L, D]
```

A vector:

```text
[D]
```

is interpreted as a single-frame sequence:

```text
[1, D]
```

Singleton leading batch dimensions are removed.

The batch result is a `TensorResourceBatch`:

```python
batch.resources["wavlm"].values
# [B, T_max, D]

batch.resources["wavlm"].lengths
# [B]
```

For resources with additional dimensions:

```text
[T, L, D]
```

collation produces:

```text
[B, T_max, L, D]
```

Trailing dimensions must agree across all samples.

## `TensorResourceBatch`

`TensorResourceBatch` stores:

```python
values
lengths
```

Indexing it returns an individual tensor trimmed to its original time length:

```python
wavlm = batch.resources["wavlm"]

sample_features = wavlm[0]
```

so padding does not need to be manually removed.

## Token sequences

`token_ids` use sequence-specific collation.

For:

```python
[12, 43, 9]
[71, 2]
```

the collated representation contains padded values and original sequence lengths.

This allows token sequences to use the same `TensorResourceBatch` interface as other variable-length tensor resources.

---

# Manifest datasets

`ManifestDataset` constructs dataset rows from one or more CSV manifests.

A basic manifest may contain:

```csv
utt_id,path,split
84-121123-0001,/data/84-121123-0001.flac,train
84-121123-0002,/data/84-121123-0002.flac,train
```

and can be loaded using:

```python
from quick_convert.data import ManifestDataset

dataset = ManifestDataset(
    manifest_path="manifest.csv",
)
```

Column names can be customized:

```python
dataset = ManifestDataset(
    manifest_path="manifest.csv",
    path_column="audio_path",
    utt_id_column="id",
    split_column="partition",
)
```

Multiple manifests may be supplied:

```python
dataset = ManifestDataset(
    manifest_path=[
        "train.csv",
        "dev.csv",
    ],
)
```

## Resources stored directly in a manifest

Manifest columns can also become resources.

For example:

```csv
utt_id,path,split,transcript
001,/data/001.wav,train,hello world
```

can be configured with:

```python
dataset = ManifestDataset(
    manifest_path="manifest.csv",
    resources={
        "transcript": {
            "column": "transcript",
            "kind": "text",
        }
    },
)
```

The resulting sample contains:

```python
sample.resources.transcript.value
# "hello world"
```

Manifest resources and provider-generated resources use the same `ResourceRef` representation and can therefore be consumed uniformly downstream.

---

# Creating dataloaders

Datasets provide a convenience method for creating their corresponding PyTorch dataloader:

```python
loader = dataset.make_dataloader(
    batch_size=32,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
)
```

This automatically uses the dataset's resource-aware `collate_fn`.

Equivalent manual construction would require supplying that collator explicitly, so `make_dataloader()` is generally preferred.

For example:

```python
for batch in loader:
    print(batch.waveforms.shape)
    print(batch.resources["wavlm"].values.shape)
```

---

# Fixed-length padding

`BaseDataset` optionally accepts:

```python
max_length = ...
```

expressed in audio samples after any configured resampling.

This forces collated waveform batches to reach a consistent maximum time dimension.

Likewise, tensor resource references can carry their own `max_length`.

This is primarily useful for workloads that benefit from stable tensor shapes, such as certain cuDNN benchmarking or compilation configurations.

The original sequence lengths remain available separately and should be used for masking.

---

# Example: audio + speaker metadata + precomputed SSL features

```python
from quick_convert.data import BaseDataset
from quick_convert.data.resources import (
    PathResourceProvider,
    TemplateResourceProvider,
)

speaker_provider = TemplateResourceProvider(
    name="speaker_id",
    template="{path.parent.parent.name}",
    kind="text",
)

wavlm_provider = PathResourceProvider(
    name="wavlm",
    path_template="/features/wavlm/{sample.utt_id}.pt",
    kind="torch_tensor",
)

dataset = BaseDataset(
    root="/datasets/LibriSpeech",
    splits=["train-clean-100"],
    file_format="flac",
    utt_id_template="{path.stem}",
    resource_providers=[
        speaker_provider,
        wavlm_provider,
    ],
    load=[
        "audio",
        "wavlm",
    ],
    target_sr=16_000,
)

sample = dataset[0]

print(sample.utt_id)
print(sample.waveform.shape)
print(sample.resources.speaker_id.value)
print(sample.resources.wavlm.value.shape)
```

Create batches using:

```python
loader = dataset.make_dataloader(
    batch_size=16,
    shuffle=True,
    num_workers=8,
)

batch = next(iter(loader))

print(batch.waveforms.shape)
print(batch.resources["speaker_id"])
print(batch.resources["wavlm"].values.shape)
print(batch.resources["wavlm"].lengths)
```

---

# Example: references without loading them

A dataset can expose feature paths without materializing the features:

```python
dataset = BaseDataset(
    root="/datasets/LibriSpeech",
    splits=["train-clean-100"],
    file_format="flac",
    utt_id_template="{path.stem}",
    resource_providers=[
        wavlm_provider,
    ],
    load=False,
)
```

Then:

```python
sample = dataset[0]

sample.resources.wavlm.path
# /features/wavlm/...

sample.resources.wavlm.value
# None
```

This is useful for preprocessing pipelines or code that wants to inspect or transform resource locations itself.

---

# Example: online resources

When features should be computed dynamically:

```python
from quick_convert.data.resources import OnlineResourceProvider

wavlm_provider = OnlineResourceProvider(
    extractor=wavlm_encoder,
    name="wavlm",
)
```

For a sample:

```python
features = wavlm_provider.provide_sample(sample)
```

and for a batch:

```python
features = wavlm_provider.provide_batch(batch)
```

This keeps the distinction between the **resource being requested** and the **component used to compute it**, while allowing experiments to share the same feature extractor implementations.

---

# Design principles

## Datasets describe samples, not experiments

`BaseDataset` should remain unaware of concepts such as:

```text
speaker classification
ASR
emotion recognition
SSL probing
prosody prediction
```

Experiment-specific values belong in resources.

## Providers locate or produce resources

Logic such as:

```text
speaker ID comes from this directory
feature path follows this template
transcript comes from this annotation file
```

belongs in resource providers rather than dataset subclasses.

## Loading is explicit

Associating a resource with a sample does not imply that it must immediately be loaded.

This keeps dataset access lightweight and allows each experiment to decide which resources it needs.

## Sample-level and batch-level representations are distinct

At sample level:

```text
resource name -> ResourceRef
```

At batch level:

```text
resource name -> collated value
```

This distinction allows the resource layer to preserve paths, metadata, and lazy-loading information before batching while presenting convenient model-ready objects afterward.

## Specialized dataset subclasses should remain thin

A new dataset subclass is useful when the dataset itself has a genuinely different indexing or discovery mechanism, such as a manifest.

Dataset-specific annotations or sidecar features generally do not require subclasses; they should instead be represented through resource providers.

---

# Public API

The commonly used data types should be importable from:

```python
from quick_convert.data import (
    AudioBatch,
    AudioSample,
    BaseDataset,
    ManifestDataset,
    MetadataSample,
)
```

Resource functionality is available from:

```python
from quick_convert.data.resources import (
    BaseResourceProvider,
    CSVTranscriptProvider,
    PathResourceProvider,
    ResourceCollection,
    ResourceRef,
    TemplateResourceProvider,
    collate_resources,
    load_resource,
)
```

Internal loader and collation helpers generally do not need to be imported by downstream experiments.

---

# Extending the data module

## Adding a resource provider

Implement a provider when resource resolution requires reusable logic beyond an existing template or path provider.

Reference-based providers should conceptually implement:

```python
class MyProvider(BaseResourceProvider):
    def __call__(self, sample) -> ResourceRef: ...
```

The provider should determine **which resource belongs to the sample**, while loading and batching remain the responsibility of the resource subsystem.

## Adding a serialized resource type

To support a new serialized representation:

1. Add a loader for the representation.
2. Register it with the resource loading system.
3. Add corresponding collation behavior if the existing text/tensor/token collators are not sufficient.

The resource's semantic purpose does not require a new dataset implementation.

## Adding a dataset

Subclass `BaseDataset` when samples are discovered through a genuinely different indexing source.

The subclass should preferably construct `MetadataSample` rows and delegate ordinary loading, provider handling, and collation back to `BaseDataset`.

`ManifestDataset` is the primary example of this pattern.

# Quickstart: Training VQ-ASR on LibriSpeech

This quickstart walks through the public VQ-ASR training pipeline. It assumes that LibriSpeech is already available locally and introduces the core concepts used throughout the codebase: Hydra configuration, datasets, resources, and data loading.

## Exploring Hydra configs

The codebase relies heavily on Hydra configuration composition. Before continuing, we strongly recommend installing the VS Code extension Hydralance:

[Hydralance](https://marketplace.visualstudio.com/items?itemName=imagirom.hydralance&utm_source=chatgpt.com)

Hydralance makes it much easier to navigate the configuration tree by providing features such as.

Throughout this quickstart, configuration fragments will reference files from `configs/`. Being able to jump directly to those files makes it much easier to understand how experiments are assembled.


## Defining a run

Experiments in `quick-convert` are configured using Hydra. A run is built by composing a collection of reusable configuration groups.

A minimal VQ-ASR run begins with:

```yaml
# @package _global_

defaults:
  - /global: default
  - /pipeline: training
  - /trainer: vq_asr
  - /components/ssl@trainer.module.online_encoders.content: w2vbert
```

These defaults establish the high-level structure of the experiment:

* `/global: default` loads project-wide defaults.
* `/pipeline: training` selects the training pipeline.
* `/trainer: vq_asr` selects the VQ-ASR training module and associated training configuration.
* `/components/ssl@trainer.module.online_encoders.content: w2vbert`

At this point, the run defines *how* training will happen, but not *which data* will be used.

## Selecting a dataset

Datasets are injected into specific slots in the configuration. To train on LibriSpeech, add:

```yaml
defaults:
  - /dataset@train_dataset: librispeech
```

The `train_dataset` slot can be filled by any dataset implementation. The training pipeline itself does not depend on LibriSpeech specifically.

You can limit to any set of splits like so:

```yaml
train_dataset:
  splits:
    - train-clean-100
```

This tells the dataset to enumerate samples from the `train-clean-100` partition only. You can include other splits like so:


```yaml
train_dataset:
  splits:
    - train-clean-100
    - train-clean-360
```

## Dataset roots

The location of the underlying data is configured separately from the training run.

Project-wide paths can be defined in `configs/global/default.yaml`:

```yaml
data_root: /path/to/data
```

Individual datasets derive their own roots from this shared location. For example, the LibriSpeech dataset may define:

```yaml
root: ${data_root}/librispeech/LibriSpeech
```

This value can also be overridden directly in a run configuration if needed.

```yaml
train_dataset:
  root: /my/custom/libri/root
  splits:
    - train-clean-100
    - train-clean-360
```


## Resources

The dataset we've defined is sufficient for loading the audio samples from Librispeech. However, we're training an ASR model, which requires paired transcripts. Any extraneous information relating to data samples is refered to as **resources**.

Resources include:

* annotations already present in the dataset, such as transcripts, speaker IDs, or emotion labels;
* precomputed artifacts, such as SSL features, token IDs, or speaker embeddings.

Resources are provided by **resource providers**. A dataset config should define resource providers for its annotations, but for custom precomputed features, you'll have to define your own. We're only dealing with annotations so far.

For example:

```yaml
train_dataset:
  splits:
    - train-clean-100

  resource_providers:
    - ${train_dataset.transcript_provider}
```

Resource providers may reference information already available in the dataset, such as file paths or split names.

LibriSpeech defines its transcript provider as:

```yaml
transcript_provider:
  _target_: quick_convert.data.resources.CSVTranscriptProvider
  path_template: "{path.parent}/{path.parent.parent.name}-{path.parent.name}.trans.txt"
  utterance_key: path.stem
  key_column: 0
  text_column: 1
  delimiter: " "
  join_text_columns: true
```

and its speaker-ID provider as:

```yaml
spkid_provider:
  _target_: quick_convert.data.resources.TemplateResourceProvider
  kind: text
  name: spkid
  template: "{path.parent.parent.name}"
```

The training pipeline will access these resources by name, defined in the `name` field.

## Loading resources

Defining a resource provider makes a resource available, but it does not necessarily load the resource into memory.

The `load` field determines which resources are materialized when a sample is retrieved:

```yaml
train_dataset:
  load:
    - audio
    - transcript
```

In this example:

* `audio` loads the waveform from disk; The audio name is available in all datasets -- it's not defined in `resource_providers`.
* `transcript` loads the transcript associated with the sample, defined above 

The names in `load` correspond to the names defined by resource providers.

Conceptually:

```text
dataset
    ↓
discovers samples

resource providers
    ↓
attach resources

load
    ↓
selects which resources are materialized
```

This distinction is especially useful for precomputed features. A provider may attach a reference to a tensor on disk, while `load` determines whether that tensor should actually be read during dataset iteration.

Setting

```yaml
load: all
```

loads audio together with every resource exposed by the configured providers.

Setting

```yaml
load: false
```

leaves all resources as references and loads only metadata.

## Launching training

Once the dataset and trainer have been configured, training can be launched from the command line:

```bash
uv run -m quick_convert.cli.train --config-name run/train_vq_asr_librispeech
```

Hydra will compose the configuration from the files listed in the `defaults` section, instantiate the dataset and training pipeline, and begin training.

Configuration values can be overridden directly from the command line. For example, to adjust the batch size:

```bash
uv run -m quick_convert.cli.train --config-name run/train_vq_asr_librispeech pipeline.batch_size=64
```

Or to change the training split:

```bash
uv run -m quick_convert.cli.train --config-name run/train_vq_asr_librispeech train_dataset.splits=[train-clean-360]
```

The next section will introduce feature precomputation and manifest construction, allowing you to build training runs from scratch rather than relying on existing artifacts.

--

# Part 2: Precomputing features and using manifests 

The VQ-ASR example in the previous section relied on LibriSpeech directly and computed all features online during training. While this keeps the configuration simple, it quickly becomes impractical for larger experiments.

First, many feature extractors are expensive to run. Self-supervised speech models such as W2V-BERT may contain hundreds of millions of parameters, and repeatedly evaluating them for every epoch can dominate training time. Precomputing these representations once allows experiments to run much faster and makes results easier to reproduce.

Second, experiments often require information that is not naturally stored in the original dataset. A project may combine multiple datasets, attach precomputed speaker embeddings, add tokenized transcripts, or store train–validation splits generated by a preprocessing pipeline.

To support these workflows, `quick-convert` provides two mechanisms:

* **Feature precomputation**, which computes representations once and stores them on disk.
* **Manifest datasets**, which describe a collection of samples in a portable CSV format.

A manifest dataset is simply a table in which each row corresponds to a sample. In addition to the audio path, manifests can store metadata such as speaker identities, transcripts, and dataset splits.

For example:

```csv
utt_id,path,split,spk_id,transcript
84-121123-0001,/data/librispeech/train-clean-100/84/121123/84-121123-0001.flac,train-clean-100,84,THE TRANSCRIPT
174-50561-0004,/data/librispeech/train-clean-100/174/50561/174-50561-0004.flac,train-clean-100,174,ANOTHER TRANSCRIPT
```

Unlike dataset-specific implementations such as LibriSpeech, manifest datasets are completely independent of the original directory structure. This makes them useful for sharing experiments, merging datasets, and attaching new resources generated during preprocessing.

The following sections explain how to generate manifests and precompute features so that they can be consumed by the training pipeline.

## Building a manifest

A manifest captures the samples and metadata exposed by a dataset in a flat CSV file.

A manifest does not replace the dataset abstraction. Instead, the manifest-building pipeline iterates over an existing dataset object and writes selected fields from each sample into a CSV file.

### Configure the source dataset

To build a manifest from LibriSpeech, begin with a run config that selects the manifest-building pipeline and the LibriSpeech dataset. Here, we're using `configs/run/build_manifest_libri.yaml`:

```yaml
# @package _global_

defaults:
  - /global: default
  - /pipeline: build_manifest
  - /dataset: librispeech
  - _self_
```

The source split can then be selected in the same way as in the previous training example:

```yaml
dataset:
  splits:
    - train-clean-100
    - train-clean-360
```

This constructs a LibriSpeech dataset over the `test-other` split.

### Attach the desired resources

The manifest can include resources exposed by the dataset’s resource providers. To include transcripts and speaker IDs, attach the corresponding providers:

```yaml
dataset:
  splits:
    - train-clean-100
    - train-clean-360

  resource_providers:
    - ${dataset.transcript_provider}
    - ${dataset.spkid_provider}
```

The LibriSpeech dataset config defines how these resources are derived from its directory structure. By the time the manifest pipeline receives each sample, the sample therefore contains:

* its utterance ID;
* its audio path;
* its split;
* a transcript resource;
* a speaker-ID resource.

### Select the manifest columns

The `columns` mapping determines the CSV schema.

Each key becomes a column name, and each value is a template evaluated against the current sample:

```yaml
columns:
  utt_id: "{sample.utt_id}"
  path: "{sample.path}"
  split: "{sample.split}"
  transcript: "{sample.resources.transcript.value}"  # this is how you access resources on a data sample.
  spkid: "{sample.resources.spkid.value}"
```

For every sample in the dataset, the pipeline evaluates these templates and writes the result as one CSV row.

The resulting manifest will have the form:

```csv
utt_id,path,split,transcript,spkid
6930-75918-0000,/path/to/LibriSpeech/test-other/6930/75918/6930-75918-0000.flac,test-other,THE TRANSCRIPT TEXT,6930
```

The manifest schema is not fixed. You may add, remove, or rename columns by changing the mapping.

For example, a minimal audio-only manifest could use:

```yaml
columns:
  utt_id: "{sample.utt_id}"
  path: "{sample.path}"
  split: "{sample.split}"
```

### Choose the output behavior

The pipeline will normally refuse to overwrite an existing manifest. To replace an existing file, set:

```yaml
pipeline:
  overwrite: true
```

The complete run config so far is:

```yaml
# @package _global_

defaults:
  - /global: default
  - /pipeline: build_manifest
  - /dataset: librispeech
  - _self_

columns:
  utt_id: "{sample.utt_id}"
  path: "{sample.path}"
  split: "{sample.split}"
  transcript: "{sample.resources.transcript.value}"
  spkid: "{sample.resources.spkid.value}"

pipeline:
  overwrite: true

dataset:
  splits:
    - train-clean-100
    - train-clean-360

  resource_providers:
    - ${dataset.transcript_provider}
    - ${dataset.spkid_provider}
```

When the run executes, `BuildManifestPipeline`:

1. iterates over the configured dataset;
2. evaluates each column template against the current sample;
3. writes one row per sample;
4. creates the output directory when necessary;
5. refuses to replace an existing file unless `overwrite` is enabled.

### Pipeline outputs

Every pipeline is responsible for producing artifacts on disk. For this reason, pipeline configurations typically define an output directory and one or more output files.

For the manifest-building pipeline, these defaults are defined in `configs/pipeline/build_manifest.yaml`:

```yaml
_target_: quick_convert.pipelines.build_manifest.BuildManifestPipeline

dataset: ${dataset}
columns: ${columns}

out_path: ${out_root}/${hydra:runtime.choices.dataset}${suffix}/manifest.csv
out_dir: ${out_root}/${hydra:runtime.choices.dataset}${suffix}
```
Note: `${hydra:runtime.choices.dataset}` is the Hydra way of accessing the name of the dataset config. IN this case, it'll evaluate to `librispeech`.
The output path is assembled from several Hydra variables:

* `out_root`, the root directory for generated artifacts;
* `hydra:runtime.choices.dataset`, the selected dataset configuration;
* `suffix`, an optional user-defined suffix.

You can name it after the split:

```yaml
split: train-clean-100
suffix: /${split}
```

But in our case where multiple slits are used, a descriptive experiment name might be best:

```yaml
dataset:
  splits:
    - train-clean-100
    - train-clean-360

...

suffix: /clean_460
```

Notice that is begins with a `/`. This is to specify that `${suffix}` should its own folder.

produces the manifest:

```text
outputs/librispeech/train-clean-100/manifest.csv
```

Most pipelines in `quick-convert` follow this pattern: the pipeline configuration specifies sensible default locations for its outputs, while run configurations customize them through variables such as `suffix`.

## Precomputing features

The previous example computed every input feature online during training. This is convenient for small experiments, but it can become expensive when the feature extractor is a large neural network or when the same features are reused across many runs.

The precomputation pipeline evaluates a feature extractor once over a dataset and saves its output to disk. Training can then load the saved features directly instead of repeatedly running the extractor.

In this example, we will precompute transcript token IDs for LibriSpeech.

### Training a SentencePiece tokenizer

Before token IDs can be precomputed, we need a trained tokenizer model.

`quick-convert` trains SentencePiece tokenizers through the same pipeline interface used for model training. The run configuration selects the generic training pipeline, a tokenizer trainer, and a dataset containing transcript resources.

Create `configs/run/train_bpe_tokenizer_librispeech.yaml`:

```yaml
# @package _global_

defaults:
  - /global: default
  - /trainer: tokenizer
  - /pipeline: training
  - /dataset@train_dataset: librispeech
  - _self_

dataset: ${train_dataset}
out_dir_suffix: _${trainer.module.vocab_size}_tokens

train_dataset:
  splits:
    - train-clean-100
    - train-clean-360
    - train-other-500

  load: false

  resource_providers:
    - ${dataset.transcript_provider}

val_dataset: null
```

This configuration trains the tokenizer over the transcripts from the three LibriSpeech training splits.

The transcript provider is attached to `train_dataset`, making a `transcript` resource available for every sample. The audio itself is not needed, so the dataset uses:

```yaml
load: false
```

The tokenizer trainer reads transcript values directly from the dataset resources rather than loading waveforms.

#### 1. Configure the tokenizer

The tokenizer trainer is defined in `configs/trainer/tokenizer.yaml`:

```yaml
_target_: quick_convert.pipelines.training.sentencepiece_trainer.TokenizerTrainer

module:
  _target_: quick_convert.pipelines.training.modules.tokenizer.bpe.SentencePieceBPETrainer

  vocab_size: 1000
  character_coverage: 1.0

  pad_id: 0
  unk_id: 1
  bos_id: 2
  eos_id: 3

  user_defined_symbols: []
  input_sentence_size: 0
  shuffle_input_sentence: true
  num_threads: 4

model_prefix: tokenizer
text_key: transcript
```

The trainer expects each sample to expose a resource named by `text_key`:

```yaml
text_key: transcript
```

For each item in the training dataset, it reads:

```python
item.resources["transcript"].value
```

and passes the resulting stream of sentences to SentencePiece.

The tokenizer configuration above creates a BPE vocabulary containing 1,000 tokens. It also reserves the following IDs:

```text
0  padding
1  unknown
2  beginning of sequence
3  end of sequence
```

These IDs should remain consistent with any downstream model that consumes the generated token IDs.

#### 2. Specify the Output location

The run config derives an output suffix from the configured vocabulary size:

```yaml
out_dir_suffix: _${trainer.module.vocab_size}_tokens
```

With a vocabulary size of `1000`, the tokenizer is therefore written to a directory ending in:

```text
_1000_tokens
```

The trainer uses:

```yaml
model_prefix: tokenizer
```

so the generated SentencePiece files include:

```text
tokenizer.model
tokenizer.vocab
```

With the current project output configuration, the model used by the following precomputation step is expected at:

```text
outputs/tokenizer/librispeech_1000_tokens/tokenizer.model
```

#### 3. Run tokenizer training

Launch the run with:

```bash
uv run -m quick_convert.cli.run --config-name run/train_bpe_tokenizer_librispeech
```

The training pipeline instantiates `TokenizerTrainer`, which iterates over the dataset transcripts and delegates training to `SentencePieceBPETrainer`.

Once this command completes, the resulting `tokenizer.model` can be used to precompute token IDs for LibriSpeech.

### Using the Tokenizer to Precompute Token Ids

Create a run configuration such as `configs/run/precompute_tokens_librispeech.yaml`:

```yaml
# @package _global_

defaults:
  - /global: default
  - /pipeline: precompute_features
  - /dataset: librispeech
  - /components/feature_extractor@feature_extractor: tokenizer
  - _self_
```

These defaults select:

* the global project configuration;
* the feature-precomputation pipeline;
* the LibriSpeech dataset;
* the tokenizer feature extractor.

The `@feature_extractor` package override places the tokenizer configuration in the `feature_extractor` slot expected by the pipeline. The prefix tells the config to look inside the `/components/feature_extractor` folder for the `tokenizer.yaml` file.

#### 1. Configure the source dataset

Select the LibriSpeech splits whose transcripts should be tokenized. In this case, we're doing all of `train` and `dev`:

```yaml
dataset:
  splits:
    - train-clean-100
    - train-clean-360
    - train-other-500
    - dev-clean
    - dev-other

  resource_providers:
    - ${dataset.transcript_provider}

  load: false  # we don't need to load audio, and the transcript resource/(annotation) is "loaded by default"
```

The tokenizer requires each sample’s transcript, so the LibriSpeech transcript provider is attached to the dataset.

Here, `load: false` prevents the dataset from eagerly materializing its resources. The extractor can request the information it needs while processing each batch.

The validation splits are included because validation runs also require precomputed token IDs.

#### 2. Configure the tokenizer

Point the extractor to the tokenizer model that should be used:

```yaml
feature_extractor:
  encoder:
    model_file: outputs/tokenizer/librispeech_1000_tokens/tokenizer.model
```

The extractor reads each sample’s transcript and converts it into the representation produced by this tokenizer.

Because the tokenizer is part of the precomputation configuration, the resulting features are tied to that tokenizer model. Changing the tokenizer requires regenerating the token IDs.

#### 3. Configure the output directory

The default precomputation pipeline configuration is defined in `configs/pipeline/precompute_features.yaml`:

```yaml
_target_: quick_convert.pipelines.precompute_features.PrecomputeFeaturesPipeline

dataset: ${dataset}
extractor: ${feature_extractor}

out_dir: ${out_root}/precomputed/${hydra:runtime.choices.dataset}/${hydra:runtime.choices.feature_extractor}

batch_size: 8
num_workers: 4
skip_existing: true
```

As with the manifest-building pipeline, the output directory is part of the pipeline configuration.

By default, it is derived from:

* `out_root`; this is defined by default as `outputs` relative to the project directory.
* the selected dataset;
* the selected feature extractor.

The token-precomputation run overrides this path:

```yaml
pipeline:
  out_dir: ${out_root}/precomputed/${hydra:runtime.choices.dataset}/tokenizer
```

Assuming the default output root is `outputs`, the generated files will be written beneath:

```text
outputs/precomputed/librispeech/tokenizer/
```

##### Generated files

The pipeline creates one PyTorch file per utterance. Files are organized by dataset split:

```text
outputs/precomputed/librispeech/tokenizer/
├── manifest.jsonl
├── train-clean-100/
│   ├── 19-198-0000.pt
│   └── ...
├── train-clean-360/
│   └── ...
├── train-other-500/
│   └── ...
├── dev-clean/
│   └── ...
└── dev-other/
    └── ...
```

For every batch, the pipeline:

1. calls `extractor.extract_batch(batch)`;
2. verifies that the extractor returned one output per sample;
3. saves each output as a `.pt` file;
4. records its location in `manifest.jsonl`.

Each manifest row has the following general form:

```json
{
  "utt_id": "19-198-0000",
  "path": "/path/to/LibriSpeech/train-clean-100/19/198/19-198-0000.flac",
  "split": "train-clean-100",
  "feature_path": "outputs/precomputed/librispeech/tokenizer/train-clean-100/19-198-0000.pt"
}
```

When available directly on the sample, the speaker ID is also added to the row.

#### 4. Specify Training Args: Batching and existing files

The pipeline exposes several practical options:

```yaml
pipeline:
  batch_size: 8
  num_workers: 4
  skip_existing: true
```

`batch_size` controls how many samples are sent to the extractor at once, while `num_workers` controls dataset-loading workers.

When `skip_existing` is enabled, an utterance whose output file already exists is not recomputed. This is useful when restarting an interrupted preprocessing run.

#### 6. Run the `precompute_features` Pipeline

The complete tokenizer precomputation run is:

```yaml
# @package _global_

defaults:
  - /global: default
  - /pipeline: precompute_features
  - /dataset: librispeech
  - /components/feature_extractor@feature_extractor: tokenizer
  - _self_

dataset:
  splits:
    - train-clean-100
    - train-clean-360
    - train-other-500
    - dev-clean
    - dev-other

  resource_providers:
    - ${dataset.transcript_provider}

  load: false

feature_extractor:
  encoder:
    model_file: outputs/tokenizer/librispeech_1000_tokens/tokenizer.model

pipeline:
  out_dir: ${out_root}/precomputed/${hydra:runtime.choices.dataset}/tokenizer
```

Run it using the same CLI entry point used for the other pipelines:

```bash
uv run -m quick_convert.cli.run \
    --config-name run/precompute_tokens_librispeech
```

After the pipeline completes, the token IDs can be attached to another dataset as a precomputed resource rather than regenerated during training.


## Adding precomputed features to the manifest

The above section on building manifests treated token_ids as an independent resource, although this resource path could could be added to the manifest, so that you would not need a separate resource provider in the training config.
The feature-precomputation pipeline and manifest-building pipeline can be combined. After precomputing token IDs, add a column containing the path to each saved tensor:

```yaml
resources_path: ${out_root}/precomputed/${hydra:runtime.choices.dataset}

columns:
  utt_id: "{sample.utt_id}"
  path: "{sample.path}"
  split: "{sample.split}"
  transcript: "{sample.resources.transcript.value}"
  spkid: "{sample.resources.spkid.value}"
  token_ids: "${resources_path}/tokenizer/{sample.split}/{path.stem}.pt"
```

The precomputation pipeline stores tokenizer outputs using the following layout:

```text
outputs/precomputed/librispeech/tokenizer/
├── train-clean-100/
│   ├── 19-198-0000.pt
│   └── ...
├── train-clean-360/
│   └── ...
└── ...
```

For a sample from `train-clean-100` whose audio filename is `19-198-0000.flac`, the generated manifest value would therefore be:

```text
outputs/precomputed/librispeech/tokenizer/train-clean-100/19-198-0000.pt
```

The resulting CSV now associates each audio sample with both its original metadata and its precomputed token IDs:

```csv
utt_id,path,split,transcript,spkid,token_ids
19-198-0000,/path/to/19-198-0000.flac,train-clean-100,THE TRANSCRIPT,19,outputs/precomputed/librispeech/tokenizer/train-clean-100/19-198-0000.pt
```

A manifest dataset can later expose the `token_ids` column as a resource, allowing training to load the saved tensor instead of invoking the tokenizer again.


## Training VQ-ASR from the manifest

We now have all the pieces needed to train VQ-ASR using precomputed token IDs:

1. a trained SentencePiece model;
2. precomputed token-ID files;
3. a CSV manifest containing a `token_ids` path for each sample.

The final run uses a manifest dataset rather than the LibriSpeech-specific dataset implementation.

Create a run configuration such as `configs/run/train_vq_asr_manifest.yaml`:

```yaml
# @package _global_

defaults:
  - /global: default
  - /pipeline: training
  - /trainer: vq_asr
  - /dataset@train_dataset: manifest
  - _self_

train_dataset:
  manifest_path: outputs/librispeech/clean_460/manifest.csv

  resources:
    transcript:
      column: transcript
      kind: text

    token_ids:
      column: token_ids
      kind: token_ids

  load:
    - audio
    - token_ids

val_dataset: null
```

### 1. Select the training pipeline and model

The first three defaults select the standard training machinery and the VQ-ASR architecture:

```yaml
defaults:
  - /global: default
  - /pipeline: training
  - /trainer: vq_asr
```

The training dataset is now selected from the manifest implementation:

```yaml
- /dataset@train_dataset: manifest
```

### 2. Read resources from manifest columns

The `resources` mapping tells the manifest dataset how its CSV columns should be interpreted:

```yaml
resources:
  transcript:
    column: transcript
    kind: text

  token_ids:
    column: token_ids
    kind: token_ids
```

The resource names are `transcript` and `token_ids`. Their values are taken from the corresponding manifest columns.

The `token_ids` column contains the path to the `.pt` file produced by the precomputation pipeline. Declaring it with the `token_ids` resource kind allows the dataset to load that saved representation.

This replaces the more indirect configuration in which the token path was reconstructed with a separate `PathResourceProvider`. The manifest already contains the complete path, so there is no need to derive it again.

### 3. Select what is loaded

The model requires the audio waveform and target token IDs:

```yaml
load:
  - audio
  - token_ids
```

The transcript remains available as a resource reference, but it does not need to be materialized for VQ-ASR training because the model uses the precomputed token IDs as its targets.

### Start without validation

The tutorial has produced a single manifest rather than separate training and validation manifests. For now, validation is disabled:

```yaml
val_dataset: null
```

A later experiment could divide the samples into separate manifests and configure them independently:

```yaml
train_dataset:
  manifest_path: path/to/train.csv

val_dataset:
  manifest_path: path/to/validation.csv
```

### 4. Launch training

Run the experiment through the shared pipeline entrypoint:

```bash
uv run -m quick_convert.cli.run \
    --config-name run/train_vq_asr_manifest
```

Hydra composes the VQ-ASR trainer, the manifest dataset, and the online W2V-BERT encoder. During training:

* audio is loaded from the manifest’s `path` column;
* token IDs are loaded from the `.pt` paths in its `token_ids` column;
* W2V-BERT content features are computed online;
* the VQ-ASR model is trained to predict the precomputed token sequence.

This completes the full workflow:

```text
LibriSpeech transcripts
        ↓
train SentencePiece tokenizer
        ↓
precompute token IDs
        ↓
build a manifest containing token-ID paths
        ↓
train VQ-ASR from the manifest
```

And that's it—you've trained your first model with `quick-convert`.

Starting from raw LibriSpeech data, we've trained a tokenizer, precomputed features, assembled a manifest dataset, and finally trained VQ-ASR. Along the way, we've introduced the core abstractions that the rest of the codebase builds on: datasets, resources, pipelines, trainers, and Hydra configuration composition.

Most experiments in `quick-convert` follow exactly the same pattern. New models and datasets typically involve defining different resources and pipelines rather than learning an entirely new framework.

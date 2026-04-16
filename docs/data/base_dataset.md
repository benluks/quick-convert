# BaseDataset

`BaseDataset` is the foundation for dataset handling in this codebase. Its job is intentionally modest: find audio files, represent them consistently, and expose them through the standard PyTorch `Dataset` interface.

That modesty is a feature, not a limitation.

This class is **general by design**. It does not assume too much about how a particular corpus is organized, what metadata it has, or how speaker identity should be derived. Instead, it provides a flexible base layer that works for many simple cases, and a clean starting point for more specialized dataset classes.

---

## Core idea

At its heart, `BaseDataset` answers a simple question:

> Given either a root directory or an explicit list of paths, what are the audio files I should expose as dataset items?

Each discovered file is wrapped in an `AudioSample`:

```python
@dataclass(frozen=True)
class AudioSample:
    path: Path
    split: str | None = None
    spk_id: Optional[str] = None
```

This gives the rest of the pipeline a lightweight, predictable representation of each sample without forcing the dataset class to load waveforms, parse large metadata tables, or commit to a particular corpus schema.

---

## What the class does well

### 1. It supports two input modes

`BaseDataset` can be constructed from either:

- a **root directory** containing audio files
- an explicit iterable of **paths**

This is one of the most important pieces of its flexibility.

#### Root-based usage

This is useful when your dataset already lives in a clean directory structure and you simply want to scan for valid audio files.

```python
dataset = BaseDataset(root="data/my_corpus")
```

#### Path-based usage

This is useful when file selection has already happened somewhere else, such as:

- a CSV manifest
- a filtering step
- a train/dev/test split defined outside the filesystem
- an experiment that operates on a custom subset of files

```python
dataset = BaseDataset(paths=my_selected_paths)
```

The class enforces that you provide **exactly one** of these:

- `root` xor `paths`

That makes dataset construction explicit and avoids ambiguous behavior.

---

### 2. It optionally understands splits

If `splits` is provided, `BaseDataset` treats each split as a subdirectory under `root` and records the split name in each `AudioSample`.

```python
dataset = BaseDataset(
    root="data/some_dataset",
    splits=["train", "dev", "test"],
)
```

This assumes a structure such as:

```bash
data/some_dataset/
|-train/
|--sample1.wav
|--...
|-dev/
|-test/
```


and gives you rows like:

- `AudioSample(path=Path("full/path/to/sample1.wav"), split="train")`
- `AudioSample(path=..., split="dev")`
- `AudioSample(path=..., split="test")`

That is a small but very useful abstraction. The base class does not try to interpret what a split *means*; it simply preserves the information so downstream code can use it.

If `splits` is omitted, the dataset just scans the whole root recursively and sets `split=None`.

---

### 3. It validates file formats cleanly

The class accepts a single format or multiple formats through `file_format`, normalizes them, and validates them against `VALID_FORMATS`.

That means all of these are supported:

```python
BaseDataset(root="data", file_format="wav")
BaseDataset(root="data", file_format=".wav")
BaseDataset(root="data", file_format=["wav", "flac"])
```

This is a nice example of the class being forgiving at the interface level while still being strict internally.

If no format is provided, it falls back to all supported formats.

---

### 4. It keeps dataset items simple

`__getitem__` returns an `AudioSample`, not raw audio tensors.

That is an important design decision.

The base dataset is concerned with **describing the corpus**, not fully defining how the audio should be loaded and processed. This separation makes the class reusable across many pipelines:

- training pipelines that load audio later
- preprocessing scripts that only need paths
- metadata preparation utilities
- anonymization pipelines that preserve path structure
- evaluation code that needs filenames and speaker IDs but not immediate waveform loading

In other words, this dataset is closer to an **index of audio samples** than a full feature-loading dataset.

---

### 5. It optionally supports speaker IDs without forcing a universal rule

The `return_spkid` flag allows the dataset to populate `spk_id` in each `AudioSample`.

But the base class does **not** guess how speaker IDs should be extracted. Instead, it defines a hook:

```python
def get_spkid(self, file_path: Path):
    raise NotImplementedError(
        f"{type(self).__name__} must implement `get_spkid` when `return_spkid=True`."
    )
```

This is exactly the right level of abstraction.

Speaker ID conventions differ wildly across datasets:

- sometimes the speaker is encoded in the parent directory
- sometimes it is embedded in the filename
- sometimes it comes from a metadata file
- sometimes a “speaker” concept does not even exist in a simple one-off dataset

Rather than baking in a brittle assumption, `BaseDataset` leaves this decision to subclasses.

---

## Why it is intentionally general

A common temptation in dataset design is to make the base class “smart” by teaching it about:

- metadata CSVs
- label parsing
- speaker conventions
- split logic
- corpus-specific path structures
- waveform loading
- feature extraction

That often feels convenient at first, but it makes the base object less reusable over time.

`BaseDataset` avoids that trap.

Its responsibilities are narrow:

1. validate inputs
2. discover files
3. wrap them as `AudioSample` objects
4. expose them via the `Dataset` interface

Everything else is left open.

That means the class stays useful across very different situations instead of becoming overly tailored to one corpus.

This is what it means to say:

> **It’s general by design.**

---

## Where the flexibility shows up in practice

Because the base class is so lightweight, it works in many situations with little or no extra code.

### Example: scan a whole directory

```python
dataset = BaseDataset(root="data/speech")
```

Use this when directory structure is enough and you just want all audio files.

### Example: work with only a few selected files

```python
dataset = BaseDataset(paths=[
    "a.wav",
    "b.wav",
    "c.wav",
])
```

Use this when selection happens upstream.

### Example: preserve split names

```python
dataset = BaseDataset(
    root="data/speech",
    splits=["cookie_theft", "rainbow", "picnic"],
)
```

Use this when the folder names carry meaningful partition information that later stages should preserve.

### Example: constrain formats

```python
dataset = BaseDataset(root="data/speech", file_format=["wav", "flac"])
```

Use this when the directory contains extra files you do not want mixed in.

---

## Why you will usually subclass it

For many real datasets, paths alone are not enough.

You may need:

- dataset-specific speaker ID parsing
- metadata attached to each item
- custom split definitions not tied directly to folders
- labels, transcripts, durations, or demographic attributes
- filtering logic based on corpus rules
- sample objects richer than just `path`, `split`, and `spk_id`

That is where subclassing comes in.

The base class gives you a consistent, reusable starting point. Subclasses add the meaning that belongs to a specific dataset.

---

## The most common subclassing pattern

The simplest and most common reason to subclass `BaseDataset` is to implement `get_spkid()`.

For example, suppose your corpus stores files like this:

```text
data/
  speaker_001/
    utt1.wav
    utt2.wav
  speaker_002/
    utt3.wav
```

Then a subclass could derive the speaker ID from the parent folder:

```python
class MyCorpusDataset(BaseDataset):
    def get_spkid(self, file_path: Path) -> str:
        return file_path.parent.name
```

Now you can do:

```python
dataset = MyCorpusDataset(root="data", return_spkid=True)
```

and each `AudioSample` will include a speaker ID.

---

## Other good reasons to subclass

### 1. To enforce corpus-specific structure

A subclass can validate assumptions like:

- required split names
- fixed directory layout
- expected file naming conventions
- presence of companion metadata files

That kind of validation does not belong in the general base class, but it fits naturally in a dataset-specific subclass.

### 2. To attach richer metadata

A corpus-specific dataset might want to associate each path with:

- transcript text
- language
- gender
- session ID
- duration
- source partition
- label or target class

You might still reuse the file discovery logic from `BaseDataset`, but then enrich or replace `self.rows` with more informative sample objects.

### 3. To load from manifests instead of the filesystem alone

Some datasets are best defined by CSV, JSONL, or Kaldi-style manifests rather than directory scanning.

A subclass can:

- read the manifest
- construct rows in a corpus-aware way
- still preserve the same overall dataset interface

### 4. To override the notion of a “sample”

In some cases, one dataset item may represent more than a single audio path. For example:

- paired source/target audio
- audio plus transcript
- audio plus anonymized counterpart
- trial/enrollment pairs for speaker verification

That is beyond the scope of `BaseDataset`, but entirely reasonable in a subclass.

---

## Design philosophy: the base class is not the whole dataset story

`BaseDataset` is not trying to be the final abstraction for every dataset in the project.

It is trying to be the **lowest-level reusable abstraction** that stays broadly useful.

That means:

- it should be easy to understand
- it should be hard to misuse
- it should not impose corpus-specific ideas too early
- it should give subclasses room to define what makes their dataset special

This is generally a healthier design than building one giant dataset class that tries to handle every corpus and every metadata convention through flags and conditionals.

---

## A good mental model

A helpful way to think about this class is:

- `BaseDataset` = **generic audio file inventory**
- subclass = **dataset semantics**

The base class tells you **what files exist**.
The subclass tells you **what those files mean**.

That division keeps the codebase cleaner and makes it easier to reuse the same infrastructure across very different datasets.

---

## Summary

`BaseDataset` is valuable precisely because it does not do too much.

It gives you:

- a standard PyTorch `Dataset` interface
- recursive audio file discovery
- optional split preservation
- flexible root-based or path-based construction
- file format normalization and validation
- a simple `AudioSample` representation
- a clear hook for dataset-specific speaker ID logic

And just as importantly, it deliberately leaves space for subclasses to define:

- where speaker IDs come from
- how metadata is attached
- what a sample should contain
- what structure a specific corpus requires

So the right way to view it is not as an incomplete dataset class, but as a **general base layer** for many different dataset implementations.

That is the point:

> **It’s general by design.**

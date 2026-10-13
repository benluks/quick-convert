# Fitting reusable transformations

A fitted transform learns state from training records, then applies that state
to a record. `Indexer` already follows this lifecycle: `fit(rows)` discovers
labels and `transform(row)` resolves and encodes a label. Its existing
`encode`, `encode_tensor`, and decoding methods remain available.
`ProsodyTransform` fits normalization statistics and bin edges from
`(measurements, speaker_id)` records and transforms the same record shape.

Both expose `fit`, `transform`, `state_dict`, and `load_state_dict` through the
small structural `FittedTransform` protocol. No base class or registry is
required. Fitting algorithms and mathematical transforms stay in components;
dataset iteration, resource selection, and file persistence stay in workflows.
`Indexer(state_path=...)` restores a saved mapping at construction.

`FitPipeline` takes a dataset, transform, and output path. It passes dataset
samples unchanged unless `resources` selects an ordered tuple of loaded values.
Only the selected training dataset is used. The transform does not load files
or discover datasets, and the pipeline has no prosody/speaker-specific logic.

```yaml
# Prosody composition:
transform:
  _target_: quick_convert.components.prosody.ProsodyTransform
pipeline:
  _target_: quick_convert.pipelines.fit.FitPipeline
  dataset: ${dataset}
  transform: ${transform}
  resources: [prosody, spkid]
  out_path: outputs/prosody/stats.pt
```

The canonical runs use the same pipeline:

```bash
uv run quick-convert fit_prosody_librispeech +transform.pitch_scale=global
uv run quick-convert fit_speaker_index_librispeech
```

For prosody, the configurable component key is now `transform`, replacing
`targets`: use `+transform.bins=equal_width`, for example. Existing measurement
files and version-1 statistics artifacts are unchanged. `ProsodyTargets` and
`FitProsodyPipeline` remain compatibility aliases/adapters for older Python and
Hydra targets; new compositions use `ProsodyTransform` and `FitPipeline`.

Restore state into a matching configured component without unpickling it:

```python
from quick_convert.components.fitting import load_fitted_state
from quick_convert.data.indexing import Indexer

indexer = Indexer("{sample.resources.spkid.value}")
load_fitted_state(indexer, "outputs/precomputed/librispeech/speaker_index/state.pt")
label = indexer.transform(sample)
```

Prosody fitting uses two dataset passes with streaming statistics and bounded
uniform frame sampling for quantiles. The pipeline reloads sidecars on each pass.
Set `quantile_sample_size` and `seed` to control the bounded sample; equal-width
bins use exact extrema. Fit only training data and reuse that state for validation.
Training wrappers should skip fitting an Indexer with a configured `state_path`.
Unknown labels fail on encoding rather than extending the saved mapping.

Precompute the named acoustic sidecars before fitting:

```bash
uv run quick-convert precompute_prosody_opensmile_librispeech
uv run quick-convert fit_prosody_librispeech
```

Install the `opensmile` extra for extraction. Measurement tensors contain frame
center time in seconds, F0 in Hz, a voiced flag, RMS, and voicing score. Audio is
not normalized by the extractor. Fit runs require a training manifest with
`split` and `spkid` columns, and prosody sidecars at the configured path.

For a fresh corpus, build and split the manifest before fitting:

```bash
uv run quick-convert build_flat_manifest_libri
uv run python -m quick_convert.cli.split_manifest --input outputs/librispeech/manifest.csv \
  --train-output outputs/librispeech/speaker_split/train.csv \
  --valid-output outputs/librispeech/speaker_split/val.csv
```

Use a within-speaker split for speaker-normalized pitch when validation speakers
need training statistics. A speaker-disjoint split requires global pitch
normalization or a separately defined policy for previously unseen speakers.

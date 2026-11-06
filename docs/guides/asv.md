# Automatic speaker verification

The ASV evaluator extracts utterance embeddings through the public `SpeakerEncoder`
interface, averages enrollment embeddings per speaker, and scores enrollment
speakers against test utterances using cosine similarity. It does not depend on
training modules or RVQ checkpoints.

## Run

Prepare two manifest CSV files with `utt_id`, `path`, and `spkid` columns. Use
non-overlapping recordings for enrollment and test to avoid evaluating a recording
against itself. Utterance IDs must be unique within each manifest; speaker IDs
must use the same naming scheme in both manifests. Audio paths must exist on the
machine running evaluation.

```bash
uv run quick-convert requirements eval_asv_librispeech
uv sync --extra cosyvoice
uv run evaluate asv_librispeech \
  enroll_dataset.manifest_path=/data/enroll.csv \
  test_dataset.manifest_path=/data/test.csv \
  pipeline.out_dir=outputs/evaluation/asv/my-run
```

The default uses the public CosyVoice CAMPPlus encoder (its model is downloaded
on first use). Override `speaker_encoder` with another public speaker encoder
configuration as needed and inspect requirements with those same overrides.
The example resamples audio to 16 kHz, as required by the default encoder.

`results.json` contains `eer` as a fraction, `eer_threshold`, and target/non-target
trial counts. An EER of 0.05 means 5%. Acceptance is `score >= threshold`. The
reported discrete EER averages FAR and FRR at the observed threshold minimizing
their difference; it does not interpolate. Equal differences select the lowest
threshold. Results from benchmarks using interpolated EER can differ.

For the metadata saved beside these results, see
[evaluation provenance](evaluation.md#evaluation-provenance).

The default protocol compares every enrolled speaker against every test utterance.
Matching speaker IDs define targets. Both target and non-target trials are
required. This is a simple all-pairs protocol, not an implementation of an
external benchmark's official trial list. Memory scales with the number of
trials.

## Python API and explicit trials

```python
from quick_convert.pipelines.evaluation.protocols.asv import ASVEvaluator
from quick_convert.pipelines.evaluation.protocols.asv.trials import ASVTrial

evaluator = ASVEvaluator(enroll_dataset, test_dataset, speaker_encoder, "outputs/asv")
results = evaluator.evaluate(
    enroll_dataset,
    test_dataset,
    trials=[
        ASVTrial(enroll_id="speaker-a", test_id="utt-1", target=True),
        ASVTrial(enroll_id="speaker-b", test_id="utt-1", target=False),
    ],
)
```

`enroll_id` identifies a speaker; `test_id` identifies an utterance. `evaluate`
returns results; `run` uses the default protocol and also writes `results.json`.

If you already have embeddings, call `evaluate_trials(enrollment, test, trials)`
from `quick_convert.pipelines.evaluation.protocols.asv`. Pass one aggregated
vector per enrolled speaker and one vector per test utterance. The scorer uses
the supplied target labels, computes cosine similarity only for listed pairs,
and returns the same EER fields as `ASVEvaluator.evaluate()`.

## Enrollment aggregation

Equal-weight averaging remains the default. To weight each recording by its
valid audio duration, excluding batch padding, pass a shared policy:

```python
from quick_convert.components.speaker import SpeakerEmbeddingAggregation

evaluator = ASVEvaluator(
    enroll_dataset,
    test_dataset,
    speaker_encoder,
    "outputs/asv-weighted",
    enrollment_aggregation=SpeakerEmbeddingAggregation("duration_weighted"),
)
results = evaluator.run()
```

For the existing CLI preset, append these overrides:

```bash
'+pipeline.enrollment_aggregation._target_=quick_convert.components.speaker.SpeakerEmbeddingAggregation' \
'+pipeline.enrollment_aggregation.method=duration_weighted'
```

Weighted enrollment requires a valid duration for every enrollment recording.
Load enrollment audio to supply lengths and sample rates. Aggregation does not
normalize or modify individual embeddings; cosine scoring owns normalization.
`extract_embeddings(..., durations=mapping)` optionally fills recording durations
while preserving its two-mapping return value.

## Optional caching

Caching is off by default. For repeated evaluations, enable it with
`+pipeline.cache_embeddings=true +pipeline.cache_key=my-model-and-data-v1`.
The key is your explicit declaration of the model weights, preprocessing, and
manifest/audio revision. Change it whenever any of those change and use a fresh
output directory. A mismatched key raises an error rather than reusing a cache.
Reusing a key after changing inputs can produce stale results. Only load cache
files you control.

When durations are requested, older caches can be enriched from loaded audio
metadata without re-encoding embeddings. Changed durations raise an error and
require rebuilding the cache. Equal-duration audio changes still require a new
cache key. Only utterances in the current dataset are returned from a cache.

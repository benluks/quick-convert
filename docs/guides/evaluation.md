# Evaluating speech outputs

Use utterance evaluation for prediction/reference metrics such as word error
rate or emotion representation distance. For enrollment and test trials, see
[automatic speaker verification](asv.md).

```bash
uv run evaluate asr_librispeech data_root=/data
```

The utterance pipeline writes per-utterance `predictions.csv` and aggregate
`results.json`. Both evaluation workflows also record provenance.

## Evaluation with a separate reference dataset

Pass `ref_dataset` to `EvalPipeline` to evaluate one recording collection
against another. Both datasets must contain the same unique `utt_id` values.
For example, give a converted recording the same `utt_id` as its original
reference, even when their paths or manifest row positions differ. Evaluation
writes predictions in prediction dataset order and raises an error on missing,
extra, or duplicate IDs. With no separate dataset, metrics read references from
the prediction batch.

### Emotion representation distance

`MeanEuclideanDistanceMetric()` compares emotion vectors and reports
`ser_med`. Provide precomputed reference vectors in the `ser_embedding`
resource, or pass `reference_systems={"ser_embedding": ser_system}` to
`EvalPipeline` with a separate reference dataset. Reuse the prediction system
as the reference system to compare vectors from the same model. Model settings
such as `device` and `model_name` belong on `OdysseySER`. Its vectors are class
probabilities. UAR requires categorical predictions and ground-truth emotion
labels and is currently unavailable.

### Evaluation provenance

Both evaluation workflows write `provenance.json` next to `results.json`. It
records the protocol, pairing policy, dataset and model descriptors, and
metrics, plus the same code revision, package version, and working directory
fields as conversion. CLI runs also include the resolved Hydra configuration
with any configured seed and dataset paths. Direct Python runs record the
information available from their supplied objects. The sidecar describes the
declared inputs and settings.

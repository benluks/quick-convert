# Continuous SSL composition → categorical duration distributions

This experimental baseline keeps the existing SSL segmentation. It assigns no
classes to content. A small neural head maps a continuous segment description
to probabilities over **integer frame-duration bins**, trained with categorical
cross-entropy. It does not model log-duration or yet generate modified speech.

## Run extraction and fitting

```bash
uv run --extra transformers quick-convert probe_duration_librispeech \
  dataset.root=/cfs/collections/librispeech/LibriSpeech \
  'dataset.splits=[dev-clean]' device=cuda
```

The default encoder is native WavLM `layer=6`, matching KNN-VC. The detector is
the same 75th-percentile per-utterance cosine cutoff with 20 ms minimum spacing,
no smoothing. Frame rate is read from the encoder; WavLM frames are 20 ms here.
One selected-layer forward pass is run per utterance in evaluation mode with no
gradients. No native frame-encoding sequences are saved.

Before extraction, the command sorts the unique speaker IDs, applies a seeded
permutation, and assigns 60%/20%/20% to training/validation/test. For the 40
speakers of complete dev-clean, this is **24/8/8 speakers**. These are experiment
partitions inside the supplied source split, not new official LibriSpeech
splits. Speaker IDs come from the LibriSpeech utterance-ID prefix; other datasets
must override `+pipeline.extractor.speaker_regex` accordingly. At least three
speakers are required, and every partition/aggregation must yield examples.

## What is represented and predicted?

For every interior segment, and overlapping pairs/triples of adjacent interior
segments within the same utterance:

1. Compute the mean vector over the original SSL frames, without interpolation.
2. Compute the population standard-deviation vector over those same frames. These are deliberately simple candidate descriptions of SSL composition,
   not an established definition of content. They discard temporal ordering.
3. Compress each vector to 128 dimensions using the same fixed, seeded Gaussian
   random projection. The projection is independent of data/labels and saved
   with the cache. It becomes identity if the requested sketch dimension is at
   least the encoder's feature dimension.
4. Keep integer native frame count as the **target only**. Input descriptors
   contain no frame counts, timestamps, total path length, speaker IDs, aggregate
   sizes or utterance IDs.

These moments omit explicit frame count, but can retain finite-sample effects
and timing information embedded in SSL values. Old fixed-grid descriptor caches
are incompatible: rerun extraction to obtain version 2 original-frame moments. A bounded feature-resampling audit checks numerical descriptor
and prediction changes on the same SSL trajectories resampled at 0.5x, 1.5x and
2x frame counts. These are feature-space copies, not new audio or natural timing
examples. Audit results are diagnostics, not evidence of complete duration
invariance. Single-segment models report probability total-variation changes
under these copies; aggregate models have no such audit in this first version.

The command fits **separate experiments for aggregation sizes 1, 2 and 3**.
Each fits its own bin edges and unconditional frequencies, so gains cannot come
merely from distinguishing single segments from pairs/triples. Overlapping
aggregates cannot cross experimental partitions because the split is by speaker.

## Models and controls

For each aggregation size, train heads on mean alone, variation alone, and both.
The default is a 128-unit hidden ReLU layer with dropout 0.1, AdamW learning rate
0.001 and weight decay 0.001, up to 1,000 steps, minibatches of 256. Validation
runs every 100 steps; stop after 5 checks without improvement and restore the
best validation checkpoint. Initialization reproduces the unconditional prior,
so step zero is a valid checkpoint if conditional fitting makes validation worse.
Only training/validation tensors enter the fitting function; test scores are
computed after checkpoint selection.

Up to 12 bins are fitted from **training** duration quantiles, with integer
inclusive upper edges. Duplicate edges collapse; the report lists the actual
bins. The last bin includes the training maximum and all larger test durations.
No test durations choose bin boundaries. Normalization mean/scale are also fitted
on training descriptors only. Unconditional categorical probabilities use
training counts with add-one smoothing.

A fourth head uses both descriptors with training labels permuted **within each
training speaker**. This preserves speaker-specific duration marginals while
breaking segment-level composition/target pairing. Its validation and test labels
remain unchanged. This is a negative control, not a replacement for the baseline.
Disable with `+pipeline.train_shuffled_control=false` if needed.

The report includes held-out negative log-likelihood (NLL), multiclass Brier
score, accuracy, ten-bin confidence ECE, speaker-balanced NLL, and paired
speaker-bootstrap intervals for the speaker-balanced NLL gain over unconditional
frequencies. **Positive NLL gain means improvement.** NLL uses log probabilities;
the target itself remains a categorical frame-count bin. No ordinal loss or
within-bin frame generator is implemented yet.

Test results from this small exploratory run are not a license to repeatedly
tune on the same held-out speakers. Use validation for subsequent choices and
reserve a fresh evaluation split for stronger final claims. Eight test speakers
also limit population-level precision.

## Bounds and outputs

Sampling is uniform over eligible examples within each partition and aggregation
size, not uniform over speakers. Each aggregation retains at most 20,000 training,
5,000 validation and 5,000 test descriptors. Speaker/utterance counts and eligible
versus retained counts are reported. At 128 dimensions per moment, the default
three aggregations store at most about **88 MiB of descriptor tensors**, plus
metadata, projection and small audit tensors. No full-corpus frame encodings are
stored. Memory also includes the current utterance and encoder.

Outputs under `outputs/duration_probe/TIMESTAMP/`:

- `duration_report.json`: compact results, actual bin edges, partition speaker
  IDs, sampling counts, validation histories, held-out scores and resampling audit.
- `descriptors.pt`: bounded feature/label cache so fitting can be repeated without
  SSL inference. Content sketches and frame targets are separate tensors.
- `models/*.pt`: experimental probe checkpoints with model state, normalization,
  bin edges, projection, descriptor selection and encoder metadata. These are
  research checkpoints, not exported Quick-Convert inference systems.
- `config.yaml`: resolved run configuration.

Upload only **`duration_report.json`** for analysis. Keep the cache locally.
Existing outputs are refused instead of overwritten; resume is not supported.
If fitting fails after extraction, reuse the saved cache in a new output directory.

## Refit from the compact cache

```bash
uv run quick-convert probe_duration_cached \
  cache_path=outputs/duration_probe/YOUR_RUN/descriptors.pt \
  device=cuda
```

This needs no audio, encoder checkpoint or Transformers backend. Use `device=cpu`
to fit locally without a GPU. Changing the number of duration bins, probe size,
optimization settings or positive descriptor modes does not require re-extraction.
The supplied cache is checked for speaker/utterance partition leakage.

Examples of overrides:

```bash
# Fit changes only:
+pipeline.n_bins=8 +pipeline.max_steps=2000 +pipeline.hidden_dim=256

# Online extraction changes (require new SSL extraction):
'+pipeline.extractor.aggregation_sizes=[1]' \
+pipeline.extractor.projection_dim=256 \
+pipeline.extractor.train_capacity=40000
```

Use `pipeline.out_dir=...` to select a fresh results directory. Detector overrides
such as `+pipeline.extractor.threshold=0.9` belong to separate experiments; compare
conditional gains against each setting's own unconditional baseline, not raw NLL
across different bin definitions.

Failure to improve can mean these simple, compressed descriptors miss important
SSL structure, the probe is inadequate, or the data are insufficient. It does not
by itself reject the segmentation or the continuous conditional-duration idea.
Improvement is also not proof of a generative timing model: detector effects and
residual timing cues remain relevant. This baseline asks a narrower question:
does this continuous SSL description improve held-out duration probabilities?

## Duration error evaluation

New runs automatically score models and the unconditional baseline with:

- Modal-bin interval MAE in frames and milliseconds: zero inside the predicted
  interval, otherwise distance to its nearest edge. The final interval has no
  upper bound; broad bins can make this metric look favorable.
- Mixture-median MAE and median absolute error, in frames and milliseconds.
  Each predicted bin probability weights the empirical distribution of training
  frame counts within that bin. The median of this mixture is the point estimate.
  Correct-bin predictions may still incur point error. The tail uses observed
  training durations, so extrapolation beyond training support is unavailable.
- Ranked probability score (RPS), averaged across bin boundaries. This scores
  the full distribution and penalizes distant bin errors more than adjacent ones;
  distance is in bin order, not physical frame widths. Lower is better.

Bin boundaries remain fixed across models within each aggregation experiment.
These metrics do not change the training loss or validation-NLL checkpoint selection.
New checkpoints save training within-bin distributions, priors and speaker splits.

Evaluate an existing checkpoint without retraining or extracting SSL again:

```bash
uv run quick-convert evaluate_duration_probe \
  checkpoint_path=outputs/duration_probe/RUN/models/aggregate1_both.pt \
  cache_path=outputs/duration_probe/RUN/descriptors.pt
```

This writes `duration_evaluation.json` with validation/test model and baseline
scores. CPU is the default. The original matching descriptor cache is required.
For older checkpoints, training-only within-bin distributions are reconstructed
from that cache; their original speaker assignment cannot be independently
verified against the checkpoint. Encoder, descriptor method and projection must
match. Evaluation does not modify existing checkpoints or reports.

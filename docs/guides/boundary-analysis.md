# Online SSL boundary statistics (experimental)

Run SSL inference online and discard representations after each utterance. The
pipeline keeps streaming moments and bounded duration samples per split, layer,
metric and detector setting. It does not save embedding tensors. Decoder or
training dependencies are not needed.

Start with a small LibriSpeech pilot:

```bash
uv run --extra transformers quick-convert analyze_boundaries_librispeech \
  dataset.root=/cfs/collections/librispeech/LibriSpeech \
  'dataset.splits=[dev-clean]' device=cuda \
  +pipeline.layers='[5,11,23]' +pipeline.max_utterances=100
```

Remove `pipeline.max_utterances` to process the whole selected dataset. Omit the
`pipeline.layers` override to analyze every returned layer. `layers` indexes the
**returned layer axis**, starting at zero; for WavLM/W2V-BERT `layer=null`, index
5 is transformer block 6, because the frontend hidden state is omitted. Negative
indices count from the end. A selected-layer encoder returns one axis entry
(index 0); `encoder_layer` records its configured layer independently.

Use the same pipeline with CLAC or another supported dataset:

```bash
QUICK_CONVERT_CLAC_ROOT=/cfs/collections/clac/CLAC-Dataset \
uv run --extra transformers quick-convert analyze_boundaries_librispeech \
  dataset=clac 'dataset.splits=[max_phonation]' \
  ++dataset.load=true ++dataset.target_sr=16000 device=cuda
```

Switch encoders by overriding the target (no new architecture is required):

```bash
uv run --extra transformers quick-convert analyze_boundaries_librispeech \
  encoder._target_=quick_convert.components.ssl.W2VBertContentEncoder \
  dataset.root=/cfs/collections/librispeech/LibriSpeech device=cuda
```

The dataset sample rate must match the selected encoder. Use continuous
representations with a reported `frame_hz`; discrete tokens are rejected.

## Detector sweep

Defaults are cosine change, per-utterance cutoffs at percentiles 50, 75, 90 and
95, no smoothing, and minimum spacings 20, 40 and 80 ms. Override these with:

```bash
+pipeline.metrics='[cosine,l2]' \
+pipeline.thresholds='[0.75,0.9,0.95]' \
+pipeline.smoothing_frames='[1,3,5]' \
+pipeline.min_distance_ms='[20,40,80]'
```

A percentile is computed separately for each utterance, layer, metric and
smoothing window. These cutoffs are **not** calibrated on a shared corpus and do
not imply equal absolute change across models. Each output row includes the
effective numerical threshold. To use fixed cutoffs, set
`+pipeline.threshold_mode=absolute` and supply metric-appropriate thresholds.
Run metrics separately when their useful absolute scales differ.

Change is `1 - cosine_similarity` or Euclidean distance between adjacent frames.
Smoothing applies a centered moving average to the change signal with replicated
endpoints; window sizes must be positive odd integers. Peak detection retains
local maxima at or above the cutoff, resolves flat peaks to their earliest
frame, and enforces spacing by strength (earlier wins ties). A constant signal
has no peaks. Millisecond spacing rounds up to an integer number of frame hops.

## Outputs and interpretation

- `config.yaml`: resolved Hydra experiment configuration.
- `utterances.jsonl`: compact per-utterance and detector-setting counts, frame
  rate, actual audio duration, effective cutoff and encoder metadata.
- `segments.jsonl`: incremental segment rows with timestamps, durations, edge
  flags, left/right strengths, neighboring durations, internal L2 movement,
  path length and endpoint displacement. Set `+pipeline.write_segments=false`
  for summary-only work (utterance rows are still written).
- `summary.json` and `summary.csv`: per-setting counts, exact streaming means
  and population standard deviations, approximate duration percentiles, and
  boundaries per second. `population=interior` excludes both recording edges;
  `all` includes them. Empty populations have null statistics, not zeros.

Strengths use the chosen, smoothed detection metric; internal movement always
uses unsmoothed L2 and excludes the bounding jumps. Within-segment displacement
uses its first and last embedding frames. One-frame segments have zero internal
movement. Neighbor durations refer to the original segmentation, including edge
neighbors.

Internal boundary coordinates use `frame_index / frame_hz` from a zero origin.
They are segmentation coordinates, not measured receptive-field centers. The
last endpoint uses the actual audio duration, absorbing frontend truncation or
padding into the last **edge** segment. The pipeline rejects internal boundaries
past the audio endpoint. Compare interior statistics when studying characteristic
lengths; do not interpret edge corrections as evidence about speech units.

Inference uses evaluation mode and no gradients. Each utterance is encoded
separately to avoid padding-dependent changes; all layers share its forward
pass. `+pipeline.batch_size=8` only batches dataset loading, not encoder inference.
Memory scales with the longest utterance and its layer outputs, detector sweep
size and `+pipeline.quantile_sample_size=10000`, not dataset duration. Quantiles
use a seeded uniform priority sample, while moments include every measurement.
Segment records can be large for broad sweeps, but contain no embeddings.

Use a new `pipeline.out_dir` for each experiment. Existing result files are
refused to prevent mixing or overwriting runs. A failed run can leave partial
JSONL files; no final summary means the run did not complete. There is no resume
support yet. These statistics describe this detector; they do not establish
phoneme boundaries or plausible duration ranges by themselves. Cross-model
alignment, perturbation repeatability, and resynthesis are follow-up experiments.

## Analyze saved segment records locally

For a large `segments.jsonl`, run this CPU post-processing command beside the
file. It does not load audio, download models, or repeat SSL inference:

```bash
uv run quick-convert summarize_boundaries \
  input_dir=outputs/boundary_analysis/YOUR_RUN
```

Upload only `YOUR_RUN/analysis/analysis.json` for discussion. That report contains
per-setting duration relationships, boundary-strength quartiles, speaker coverage
and per-speaker duration summaries. `analysis.csv` and `analysis.md` provide
spreadsheet and readable versions. All processing is local; no service or upload
is invoked. Use `pipeline.out_dir=/another/new/directory` to change output location.
Existing reports are refused rather than overwritten.

The command reads one contiguous utterance/setting segmentation at a time, plus
small aggregate state per setting and speaker. Memory does not scale with the
number of segment rows; bookkeeping does retain utterance IDs to detect repeated
or noncontiguous segmentations. Keep the original file ordering. Truncated
segmentations and malformed records raise errors instead of producing a partial
report. A progress bar reports bytes read, not an estimate of model inference.

Relationships are computed separately for every detector setting:

- Segment duration versus preceding/following boundary strength.
- Segment duration versus average internal L2 movement.
- Segment duration versus the next segment's duration.

Only interior segments are used; neighbor pairs require both segments to be
interior. Each relationship includes pooled Pearson correlation and a
within-utterance rank correlation. The latter centers average percentile ranks
inside each utterance, then pools the centered pairs. Percentile scaling avoids
weighting long utterances cubically through raw rank magnitudes. It is **not**
identical to pooled Spearman correlation or a simple mean of utterance-level
Spearman coefficients. Constant variables and utterances with fewer than three
eligible pairs are excluded from the rank estimate; counts are included.

The default run config extracts LibriSpeech speaker IDs with `^([^-]+)-`.
For another ID convention, supply `pipeline.speaker_regex` with exactly one
capture group, or `pipeline.speaker_regex=null` when speaker IDs are unavailable.
The command refuses IDs that do not match the selected pattern.

Intervals use a seeded bootstrap that resamples speakers, retaining all their
utterances (default 1,000 replicates; `+pipeline.bootstrap_samples=0` disables it).
With fewer than two speakers or unavailable IDs, intervals are null. Intervals
from very few speakers should not support population-level conclusions.
Strength quartiles use within-utterance average percentile ranks, so tied values
can yield unequal or empty bins. Missing/undefined correlations are null.

These are descriptive relationships conditioned on the boundary detector.
In particular, the detector itself can induce an association between duration
and internal movement. The report does not compare layers or establish whether
a segment can be stretched naturally.

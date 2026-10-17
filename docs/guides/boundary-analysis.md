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

### Movement-budget analysis

The same post-processing command now also measures duration versus
`path_length_l2` and `net_displacement_l2`. To reanalyze an existing run without
overwriting an earlier report:

```bash
uv run quick-convert summarize_boundaries \
  input_dir=outputs/boundary_analysis/YOUR_RUN \
  pipeline.out_dir=outputs/boundary_analysis/YOUR_RUN/movement_analysis
```

Upload `movement_analysis/analysis.json`. Report schema version 2 adds a
`movement_budget` section per detector setting, containing:

- Mean and population standard deviation of total path length, endpoint
  displacement and mean internal movement.
- The same measurements in fixed duration ranges `[0,80)`, `[80,160)`,
  `[160,320)`, `[320,640)`, `[640,1280)`, `[1280,infinity)` ms. Machine-precision
  timestamp errors are rounded to six decimal places in milliseconds for bin
  assignment only.
- Within-utterance duration quartiles, reducing confounding from differences
  in typical duration between recordings. Ties can produce unequal bins.
- A within-utterance log-log slope for each total movement measure versus the
  number of internal frame transitions, with speaker-bootstrap intervals and
  inclusion counts. `movement_budget.csv` contains flat bin summaries; the
  JSON and Markdown include the slope estimates.

The scaling model is
`log(movement) = utterance_intercept + beta * log(internal_transitions)`.
Both log variables are centered within each utterance before regression. A
slope near 0 is consistent with a fixed total movement budget; a slope near 1
indicates approximately constant average movement per transition. Intermediate
slopes indicate that total movement grows, but more slowly than segment length.
Use the interval and binned summaries rather than interpreting a small duration
correlation as proof that movement is constant. The fitted model is descriptive,
not a test of equivalence to zero.

Internal transitions equal `end_frame - start_frame - 1`; one-frame segments
have no internal transitions. Log fits exclude zero movement, one-frame segments
and utterance subsets without variation in transition count. Counts for these
exclusions are reported. Zero-movement segments remain in ordinary summaries
and duration correlations. Bounding jumps remain excluded from all internal
movement measures.

These estimates compare segments that have not been matched by SSL trajectory
identity. Even a fixed total movement budget would not establish that a longer
segment is a time-stretched version of a shorter one; that requires the planned
cross-utterance trajectory-matching experiment and detector controls.

## Duration-independent SSL trajectory matching (experimental)

This next experiment requires a new online SSL pass because scalar segment
records cannot reconstruct trajectories. It retains a bounded sample of segment
descriptors in RAM and writes only compact reports and match metadata.

```bash
uv run --extra transformers quick-convert analyze_trajectories_librispeech \
  dataset.root=/cfs/collections/librispeech/LibriSpeech \
  'dataset.splits=[dev-clean]' device=cuda
```

This config selects **native WavLM `encoder.layer=6`**, matching KNN-VC's
`hidden_states[6]`. It does not use the all-layer output-axis indexing from the
boundary sweep. The encoder is called once per utterance, in eval/inference mode.
The complete dev-clean split is used unless `+pipeline.max_utterances=100` is
supplied. A capped, ordered pilot may contain very few speakers; the report
includes retained utterance and speaker coverage.

Defaults are the 75th-percentile cosine cutoff, 20 ms minimum peak spacing,
no smoothing, **2,000 retained segment/control pairs**, **256 sampled queries**,
16 points per trajectory, and 5 candidates per query for DTW refinement.
Override constructor settings using `+pipeline.max_segments=4000`,
`+pipeline.max_queries=512`, `+pipeline.points=32`, etc. One detector setting is
used per run to keep retrieval populations interpretable. To test another
setting, use a fresh output directory and change `+pipeline.threshold=0.9` or
`+pipeline.min_distance_ms=80`.

### Matching and controls

For each detected interior segment containing at least two frames:

1. Draw a random contiguous span in the same utterance with exactly the same
   frame length. Its endpoints must also be internal; it ignores detector
   peaks and can overlap other random spans or coincide with the detected span.
2. Maintain a uniform reservoir over eligible detected segments, retaining each
   segment together with its paired random-location control. This guarantees
   identical retained duration distributions and utterance origins for the two
   populations. The sampling unit is a segment, not an utterance or speaker.
3. Resample each retained trajectory to 16 uniformly spaced relative-time
   points, including its endpoints. Normalize each interpolated feature vector
   for cosine distance. Preserve absolute SSL location and temporal ordering;
   do not subtract each segment's mean. Duration never enters the descriptor.
4. Search the same-sized detected and random-location banks separately. Exact,
   batched search averages pointwise cosine distance over the fixed time grid.
   Exclude **all segments from the query utterance**, not just the query itself.
5. Rerank the top 5 candidates using DTW on retained, resampled trajectories.
   The default band is 25% of the normalized time grid. Choose the path with
   minimum accumulated cosine cost, then divide by its number of matched
   steps; this is not a minimum-average-cost DTW objective. Native-frame
   sequences are discarded, so refinement cannot resolve detail lost during
   16-point resampling.

For each population, the report compares ordinary nearest neighbors, DTW-refined
neighbors, a uniformly random eligible partner, and a nearest-neighbor search
after independently shuffling the time points in every descriptor. Shuffling
preserves sampled vectors while disrupting order. Candidate pools, query IDs
and bank sizes are the same for the ordered and shuffled searches. Nearest
neighbors beat random partners by construction; that alone does **not** validate
the matching. Compare with the shuffled-time and random-location searches,
inspect examples, and assess similarity before interpreting duration ratios.

Speaker IDs are used only to report coverage/same-speaker match rates and,
optionally, exclude same-speaker candidates with `+pipeline.different_speaker=true`.
They are not features in the distance. The default regex extracts LibriSpeech
speaker IDs; override `pipeline.speaker_regex` or set it to null for unknown IDs.
No transcripts, phonetic labels, acoustic measurements or external alignments
are used to identify similar segments.

### Outputs and resource bounds

- `trajectory_analysis.json`: small aggregate report, including similarity and
  duration-ratio distributions, fractions differing by at least 1.5x/2x, duration
  ratios within similarity quartiles, speaker coverage, and DTW warp statistics.
  It also contains paired detector/control movement summaries and pooled plus
  within-utterance rank duration relationships over **all eligible segments**,
  rather than only the retained reservoir.
- `trajectory_matches.jsonl`: per-query matches and controls with paths, frame
  coordinates, timestamps, original durations and movement measurements for
  inspection/listening. It contains no feature tensors.
- `config.yaml`: resolved run configuration.

Upload **`trajectory_analysis.json`** first. Keep match records locally unless
specific examples are needed. Query distributions are descriptive; matches
share candidates and speakers and are not independent observations. No
confidence intervals or content-equivalence claims are assigned to them.

At the default 16 points and 1,024 features, the two preallocated CPU descriptor
banks require about **250 MiB**. Working memory also includes the longest
utterance's SSL output, a shuffled-bank copy, and retrieval tensors on
`pipeline.retrieval_device` (defaults to `${device}`). Search materializes only
query-batch-by-bank score blocks; it does not retain a full all-pairs matrix.
Runtime scales with retained bank size, query count, descriptor dimension and
DTW candidate count, not all possible corpus segment pairs. Use
`pipeline.retrieval_device=cpu` for CPU retrieval. New output directories are
required; resume is not supported. Reducing bank size can change nearest
neighbors, so comparisons should use equal bank/query budgets.

Time normalization does not remove every possible duration cue: the SSL values
may themselves retain timing information, and interpolation can reveal native
sampling density. Similarity means proximity under this declared descriptor,
not proven linguistic identity. A large nearest-match duration ratio supports
further investigation; it does not demonstrate plausible rhythm modification
or natural resynthesis.

### Inspect actual retrieved pairs

After trajectory analysis, generate a small, self-contained HTML inspection
report. This re-encodes only the source utterances referenced by selected
matches and their saved random partners, not the full split:

```bash
uv run --extra transformers quick-convert inspect_trajectories \
  input_dir=outputs/trajectory_analysis/YOUR_RUN device=cuda
```

`input_dir` must contain `trajectory_analysis.json` and
`trajectory_matches.jsonl`. The default encoder is native WavLM layer 6,
matching the trajectory experiment above. For other experiments, override
`encoder._target_`, `encoder.layer` and other encoder settings consistently.
The command checks encoder metadata against the original run and verifies that
recomputed real-pair scores agree within 0.001. It fails before writing reports
if these checks fail, rather than displaying comparisons from another model
or source audio. Small device/precision differences are tolerated; investigate
larger differences before changing `+pipeline.score_tolerance`.

Open `YOUR_RUN/inspection/inspection.html` in a browser. The file embeds:

- Original pair durations, saved/recomputed cosine scores, and distance between
  the mean descriptor vectors (a separate diagnostic for static SSL proximity).
- Native-frame cosine-distance matrices, with display subsampling capped at
  128 frames per side. This cap changes only the displayed matrix, not scores.
- The full normalized-time distance matrix and pointwise distance profile on
  the original run's descriptor grid (16 points by default).
- Known 1.5x and 2x feature-stretched copies of the query as numerical references.
  These interpolate already-extracted SSL embeddings, **not audio waveforms**.
- The saved random partner's distance profile when available.
- Embedded source audio clips with 150 ms context on each side; original
  segment timestamps are printed above the plots. Listening is optional and
  does not supply labels or alignments to the SSL matching method.

All panels use an absolute cosine-distance scale of **0 to 2**: pale means
close, purple means far. A pale diagonal band indicates closer vectors at
corresponding relative times. The dashed diagonal is only an equal-relative-time
reference, not a fitted alignment. Pointwise profiles help reveal whether a
moderate average hides a large mismatch in part of the trajectory. Hover over
heatmap cells to see numerical distances. This report performs no PCA or
external linguistic/acoustic alignment.

Selection defaults to detected, ordinary nearest-neighbor matches: three
score-rank strata crossed with duration ratios below/above 1.5x, up to two pairs
per cell (at most 12). This deliberately covers the score range; it is not a
representative or independent sample. Some cells may be empty. Change
`+pipeline.pairs_per_stratum=1`, `+pipeline.score_bins=4`,
`+pipeline.duration_ratio_split=2.0`, or `+pipeline.condition=dtw_refined` as needed.
Shuffled-time records cannot be inspected as real ordered trajectories.

`inspection.json` stores the same numerical diagnostics and selected match
metadata without feature tensors or embedded audio. Upload **`inspection.html`**
to discuss the visual comparisons, or `inspection.json` for numerical review.
The HTML has no remote assets, JavaScript, or server requirement; copying the
single file preserves plots and audio. Set `+pipeline.include_audio=false` to
omit audio and reduce file size. Existing inspection reports are refused;
choose `pipeline.out_dir=/another/new/directory` to rerun.

The complete source utterance is encoded before slicing saved frame intervals,
so SSL context matches the original experiment. Only requested segment tensors
are retained in RAM until rendering; no raw embeddings are saved. If audio has
moved, use both `+pipeline.path_prefix_from=/old/root` and
`+pipeline.path_prefix_to=/new/root` for a component-aware path-prefix replacement.

A synthetic stretched copy should have a very small distance: it checks the
metric and interpolation numerics. This is an ideal self-reference, **not** a
threshold for accepting naturally occurring pairs. Closeness under the chosen
metric, even together with duration variation, does not by itself establish
that two segments realize identical content or can be stretched naturally.

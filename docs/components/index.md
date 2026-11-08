# Components

Components are reusable building blocks beneath task-level systems:

- encoders and decoders;
- feature extractors and SSL backends;
- quantizers and neural-network layers;
- losses;
- speaker encoders and generators;
- low-level acoustic features.

Components should expose explicit tensor and length contracts. They may compose recursively, but should not own datasets, output directories, training loops, or workflow policy.

## Content encoders

See [acoustic and paralinguistic encoders](acoustic-content-encoders.md) for
SPEAR, PASE+, VoiceFM, emotion2vec, setup examples, and the CARE checkpoint prerequisite.

Every `ContentEncoder` declares its required waveform `sample_rate`, its
`frame_hz` when the representation has a regular frame timebase, and a batch
entry point `forward(AudioBatch) -> ContentFeatures`. Returned
`ContentFeatures` repeat the effective `frame_hz` alongside exact valid frame
lengths so downstream tools can align representations by time rather than by
proportional sequence position. Utterance-level representations use
`frame_hz=None`.

`frame_hz` describes the spacing between successive frames. It does not imply
that a frame is centered at time zero or describe the frontend's receptive
field; consumers that require sample-exact boundaries should keep those
distinctions explicit.

Complete inference behavior belongs in [systems](../design_philosophy.md#systems); orchestration belongs in [pipelines](../design_philosophy.md#pipelines). Vendored implementations under `quick_convert.external` are private dependencies reached through first-party adapters.

See the [CosyVoice mel and vocoder contract](cosyvoice.md) for reconstruction features and checkpoint compatibility.

The [Upsample Conformer encoder](upsample-conformer.md) exposes configurable
width, depth and temporal upsampling through a first-party adapter.

For backend-specific setup and contracts, use [DAC](dac.md),
[native codecs](../guides/native-codecs.md), and the content-encoder guide above.

See [PSST1](psst1.md) for the ECAPA, speaker flow, emotion compensation, and vocoder adapters.

## Component organization

Components are grouped by computational role, then by representation where useful:

| Package | Responsibility |
| --- | --- |
| `encoders/content`, `encoders/speaker` | Audio to learned representations; speaker includes pooled and frame-wise CAMPPlus |
| `sequence` | Feature-sequence processing and temporal upsampling |
| `heads` | Prediction heads and their supervision contracts |
| `generators/speaker` | Speaker-vector generation |
| `generators/acoustic` | Acoustic representation generation, optionally composing a vocoder |
| `generators/waveform` | Waveform synthesis |
| `flow_matching` | Shared flow objectives, estimators, and integration |
| `features` | Acoustic measurements and explicitly named frontend recipes |
| `transforms` | Representation transformation, aggregation, and fitted state |
| `conditioning` | Reusable reference-prompt construction |
| `resources` | Resource computation adapters and online/cached resolution |
| `layers`, `losses`, `metrics` | Neural primitives, objectives, and measurements |

Python imports and Hydra selection groups use these canonical locations.
Old component paths and selection groups have been removed. Archived configs
must be migrated before restoring models; model weight keys are unchanged.

### Speaker output and implementation variants

Speaker encoders select `output: pooled` (default) or `output: frames`.
CAMPPlus supports both; other backends explicitly reject frames until implemented.
CAMPPlus pooled features are 192-dimensional, frame features 512-dimensional;
`POOLED_DIM` and `FRAME_DIM` expose these fixed capabilities. Frame results
retain valid lengths and the prefix encoding API.

`LayerWeightedConformerEncoder` fuses SSL layers before one Conformer stack;
`ParallelLayerWeightedConformerEncoder` learns independent layer mixtures for
parallel stacks before combining outputs. `ConformerEncoder` processes a single
feature sequence.

`GradientResidualVectorQuantizer` uses factorized, cosine-normalized codebooks
trained with gradients. `EMAResidualVectorQuantizer` uses exponential moving
average codebook updates; its underlying `EMAVectorQuantizer` remains distinct
from `FactorizedCosineVectorQuantizer`.

### Mel recipes

`features.mel.matcha.MelSpectrogramConfig` specifies the Matcha/CosyVoice
reflect-padded, Slaney log-magnitude mel recipe, including exact output lengths.
Its parameters do not describe every possible mel implementation.
`features.mel.whisper` and `features.mel.s3tokenizer` explicitly delegate to their
selected backend versions, retaining backend filters and compression. Whisper
retains its backend default mel size unless explicitly overridden.
Kaldi filterbanks used by CAMPPlus and torchaudio frontends inside vendored
models remain their own recipes; vendored implementations are unchanged.

Pooled CAMPPlus retains its backend's random crop above ten seconds; frame-wise
CAMPPlus encodes requested prefixes and aligns them to a requested frame clock.
They are separate speaker encoders, not two interchangeable pooling settings.

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

The default protocol compares every enrolled speaker against every test utterance.
Matching speaker IDs define targets. Both target and non-target trials are
required. This is a simple all-pairs protocol, not an implementation of an
external benchmark's official trial list. Memory scales with the number of
speaker/utterance pairs.

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

## Optional caching

Caching is off by default. For repeated evaluations, enable it with
`+pipeline.cache_embeddings=true +pipeline.cache_key=my-model-and-data-v1`.
The key is your explicit declaration of the model weights, preprocessing, and
manifest/audio revision. Change it whenever any of those change and use a fresh
output directory. A mismatched key raises an error rather than reusing a cache.
Reusing a key after changing inputs can produce stale results. Only load cache
files you control.

## VoicePrivacy 2026 pretrained attacker

Install the optional backend and extract the official checkpoint archive:

```bash
uv sync --extra voiceprivacy-asv
curl -L --fail -o asv_ssl.zip https://duke.app.box.com/shared/static/wo07rwi5xcgam1v6qlzfnzb536rqug14
unzip asv_ssl.zip
export VOICEPRIVACY_ASV_DIR=/absolute/path/to/asv_ssl
uv run quick-convert eval_asv_voiceprivacy_ecapa \
  enroll_dataset.manifest_path=/data/enroll.csv \
  test_dataset.manifest_path=/data/test.csv \
  pipeline.out_dir=outputs/evaluation/asv/voiceprivacy
```

Set the directory to the extracted folder containing `embedding_model.ckpt`
and `WavLM-Large.pt` (check the archive's directory layout). Alternatively,
set `speaker_encoder.ecapa_checkpoint` and `speaker_encoder.wavlm_checkpoint`
explicitly. Load only trusted checkpoints: the original WavLM format requires
Python pickle loading.

```python
from quick_convert.components.speaker import VoicePrivacyECAPAEncoder

encoder = VoicePrivacyECAPAEncoder(
    ecapa_checkpoint="exp/asv_ssl/embedding_model.ckpt",
    wavlm_checkpoint="exp/asv_ssl/WavLM-Large.pt",
    device="cuda",
)
embedding = encoder.encode_file("speech.wav")  # SpeakerEmbedding, values [192]
```

The adapter remains frozen in evaluation mode. It combines 24 WavLM-Large
layers with the official ECAPA filterbank branch and returns L2-normalized
192-dimensional embeddings. It uses peak normalization and exact edge-zero
trimming for filterbanks, sentence mean normalization, and WavLM's checkpoint
waveform normalization. Batch items are processed individually at their true
length, avoiding padding-dependent embeddings. Finite non-silent mono audio
at 16 kHz, with at least 400 samples in its nonzero span, is required; file
loading performs resampling and downmixing.

This selects the attacker only. The default QC all-pairs trials and discrete
EER remain as described above; selecting this model does not reproduce the
full VoicePrivacy protocol. Fine-tuning is not provided by this frozen adapter.
The vendored implementation is pinned to upstream commit
`d5c7f4393db3e09247f58a56ca4d340fcf1ed850`; its ECAPA code carries the upstream
GPL-3.0-or-later license, with Microsoft MIT notices on WavLM sources.

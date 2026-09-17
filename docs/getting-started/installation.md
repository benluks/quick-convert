# Installation

Quick Convert requires Python 3.11 or newer. The base installation provides data handling, core model interfaces, Hydra configuration, and the universal CLI.

## Development checkout

```bash
git clone https://github.com/benluks/quick-convert.git
cd quick-convert
uv sync --group dev
```

Run `uv run quick-convert --help` to confirm that the installed run configurations are visible.

## Optional profiles

Install only the backends required by your workflow:

| Profile | Use it for |
|---|---|
| `manifests` | CSV manifest splitting |
| `transformers` | W2V-BERT and WavLM models |
| `sentencepiece` | Tokenizer training and inference |
| `wer` | JIWER word-error rate |
| `asr` | Convenience bundle for SentencePiece and WER |
| `training` | Lightning, W&B, TensorBoard, and training plots |
| `cosyvoice` | CosyVoice reconstruction and HiFT vocoding |
| `whisper` | Whisper ASR evaluation |
| `emotion2vec` | emotion2vec features |
| `mpm` | Masked Prosody Model features |
| `espnet-wavlm-joint` | ESPnet WavLM speaker embeddings |
| `pyannote` | Experimental pyannote WeSpeaker integration |
| `dac` | Experimental Descript Audio Codec encoder |

For the VQ-ASR reference workflow:

```bash
uv sync --extra transformers --extra asr --extra training --extra manifests
```

Some profiles intentionally conflict because their upstream packages require incompatible dependency versions. See the [dependency profile notes](../maintenance/dependency_extras.md).

## What installation does not include

Optional model checkpoints and datasets are not bundled. A command may still download a model on first use; each workflow guide identifies those cases.

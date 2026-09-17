# Quickstart: train VQ-ASR on LibriSpeech

This guide follows the supported reference path from LibriSpeech to VQ-ASR training and inference export.

## Install and configure data

    git clone https://github.com/benluks/quick-convert.git
    cd quick-convert
    uv sync
    uv run quick-convert requirements train_vq_asr_librispeech
    uv sync --extra asr --extra manifests --extra training --extra transformers

The requirements command derives the training dependencies from the configured
system. This end-to-end guide additionally installs `manifests` for its CSV
splitting step. If you override the content encoder, run the requirements
command with the same Hydra overrides before installing dependencies.

The reference configs expect the standard LibriSpeech directories beneath
`/data/librispeech/Librispeech/`. For example:

    /data/librispeech/Librispeech/train-clean-100/
    /data/librispeech/Librispeech/dev-clean/

Use a different `data_root` if your `librispeech/` directory lives elsewhere.
List the installed runs with:

    uv run quick-convert --help

## Train the tokenizer and prepare token IDs

    uv run train bpe_tokenizer_librispeech data_root=/data
    uv run precompute tokens_librispeech data_root=/data

These commands write the tokenizer to
`outputs/tokenizer/librispeech_1000_tokens/` and per-utterance token IDs beneath
`outputs/precomputed/librispeech/tokenizer/`.

VQ-ASR runs W2V-BERT online, so this recipe does not precompute W2V-BERT
features. The model downloads `facebook/w2v-bert-2.0` when training starts
unless it is already cached.

## Build and split a manifest

    uv run quick-convert build_flat_manifest_libri data_root=/data
    mkdir -p outputs/librispeech/speaker_split
    uv run python -m quick_convert.cli.split_manifest \
        --input outputs/librispeech/manifest.csv \
        --train-output outputs/librispeech/speaker_split/train.csv \
        --valid-output outputs/librispeech/speaker_split/val.csv

The flat manifest contains audio paths, transcripts, speaker IDs, and split
names. The split command assigns rows from each speaker to both training and
validation; it is not speaker-disjoint.

## Train and export

    uv run train vq_asr_librispeech \
        data_root=/data \
        trainer.train_dataloader_kwargs.batch_size=16

The top-level `system` is the inference model. Lightning adds the training
objectives, logging, and optimization around it.

The command prints the path of its resolved `config.yaml`. Its parent directory
is the run directory and also contains `checkpoints/last.ckpt`. For example, if
the printed path is `outputs/vq_asr/abc123/config.yaml`, export with:

    uv run quick-convert export outputs/vq_asr/abc123 models/vq-asr

Then load the inference-only artifact:

    from quick_convert.inference import load_inference_artifact

    system = load_inference_artifact("models/vq-asr", map_location="cpu")

The artifact contains the resolved system config and weights without optimizer,
scheduler, logger, or callback state.

## Expected outputs

Before training, these files should exist:

    outputs/tokenizer/librispeech_1000_tokens/tokenizer.model
    outputs/librispeech/speaker_split/train.csv
    outputs/librispeech/speaker_split/val.csv

Token ID files should also exist beneath
`outputs/precomputed/librispeech/tokenizer/<split>/`. Model downloads and a full
training run remain environment-dependent integration work.

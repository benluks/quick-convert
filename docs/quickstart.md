# Quickstart: VQ-ASR on LibriSpeech

This guide follows the supported reference path from LibriSpeech to VQ-ASR training and inference export.

## Install and configure data

    git clone https://github.com/benluks/quick-convert.git
    cd quick-convert
    uv sync --extra transformers --extra asr --extra training --extra manifests

The reference dataset expects `/data/librispeech/Librispeech/` with the standard train and development split directories. Pass `data_root=/data` below. Installed runs live under `quick_convert/configs/run/`; list them with:

    uv run quick-convert --help

## Prepare features

    uv run train bpe_tokenizer_librispeech data_root=/data
    uv run precompute tokens_librispeech data_root=/data
    uv run precompute content_w2vbert_librispeech data_root=/data

The tokenizer defaults to `outputs/tokenizer/librispeech_1000_tokens/`. W2V-BERT downloads `facebook/w2v-bert-2.0` unless cached.

## Build and split a manifest

    uv run quick-convert build_flat_manifest_libri data_root=/data
    mkdir -p outputs/librispeech/speaker_split
    uv run python -m quick_convert.cli.split_manifest \
        --input outputs/librispeech/manifest.csv \
        --train-output outputs/librispeech/speaker_split/train.csv \
        --valid-output outputs/librispeech/speaker_split/val.csv

This assigns rows from each speaker to both train and validation; it is not speaker-disjoint.

## Train and export

    uv run train vq_asr_librispeech \
        data_root=/data \
        trainer.train_dataloader_kwargs.batch_size=16

The top-level `system` is a plain `VQASRSystem`; Lightning wraps it with objectives, logging, and optimization.

    uv run quick-convert export outputs/my-run models/vq-asr

Then load the inference-only artifact:

    from quick_convert.inference import load_inference_artifact

    system = load_inference_artifact("models/vq-asr", map_location="cpu")

The artifact contains the resolved system config and weights without optimizer, scheduler, logger, or callback state. The fast suite composes every installed run; full downloads and training remain environment-dependent integration work.

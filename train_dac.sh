#!/bin/bash
#SBATCH --job-name=dac_train
#SBATCH --partition=gpu           # adjust to your Triton GPU partition (see `sinfo -s`)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ---------------------------------------------------------------------------
# GPU training of the controllable-RVQ anonymiser on precomputed DAC features.
# PREREQS (on the LOGIN node):
#   1. All features already precomputed (content/emo2vec/spk) + manifest + tokenizer.
#   2. Create the training env once:
#        UV_PROJECT_ENVIRONMENT=.venv-train uv sync --extra module-training --extra chatterbox
#      (add more extras if instantiation reports a missing SSL import)
# Then: sbatch slurm/train_dac.slurm
# ---------------------------------------------------------------------------

set -euo pipefail
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"

export UV_PROJECT_ENVIRONMENT=.venv-train
export HF_HOME="${WRKDIR:-$HOME}/hf_cache"

uv run --offline -m quick_convert.cli.train \
    --config-name run/train_controllable_rvq_dac_librispeech

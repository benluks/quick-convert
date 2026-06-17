#!/bin/bash
#SBATCH --job-name=dac_train
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu-debug
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/elec/t412-speechsynth/palp/vpc_collab/scripts/logs/trainwdac_%j.out
#SBATCH --error=/scratch/elec/t412-speechsynth/palp/vpc_collab/scripts/logs/trainwdac_%j.err

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
# Auth via `wandb login` on the login node (writes ~/.netrc) — no key in this file.

OUT=/scratch/elec/t412-speechsynth/palp/vpc_collab/inesc-gitlab/quick-convert/outputs

uv run --offline -m quick_convert.cli.train \
    --config-name run/train_controllable_rvq_dac_librispeech_v2 \
    out_root=$OUT \
    +pipeline.train_kwargs.num_sanity_val_steps=0 \
    +pipeline.train_kwargs.limit_val_batches=0
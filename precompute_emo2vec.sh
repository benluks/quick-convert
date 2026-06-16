#!/bin/bash
#SBATCH --job-name=emo2vec
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu-a100-80g           
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/elec/t412-speechsynth/palp/vpc_collab/scripts/logs/pre_emo2vec_%j.out
#SBATCH --error=/scratch/elec/t412-speechsynth/palp/vpc_collab/scripts/logs/pre_emo2vec_%j.err

# PREREQS (LOGIN node):
#   UV_PROJECT_ENVIRONMENT=.venv-emo uv sync --extra conditional-rvq
#   UV_PROJECT_ENVIRONMENT=.venv-emo HF_HOME=$WRKDIR/hf_cache \
#     uv run python -c "from funasr import AutoModel; AutoModel(model='iic/emotion2vec_plus_large')"

set -euo pipefail
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"
export UV_PROJECT_ENVIRONMENT=.venv-emo
export HF_HOME="${WRKDIR:-$HOME}/hf_cache"

uv run --offline -m quick_convert.cli.precompute \
    --config-name run/precompute_emo_emo2vec_librispeech

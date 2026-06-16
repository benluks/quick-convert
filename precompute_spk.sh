#!/bin/bash
#SBATCH --job-name=spk_espnet
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=gpu-v100-32g
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/elec/t412-speechsynth/palp/vpc_collab/scripts/logs/prec_spk_%j.out
#SBATCH --error=/scratch/elec/t412-speechsynth/palp/vpc_collab/scripts/logs/prec_spk_%j.err

# PREREQS (LOGIN node):
#   UV_PROJECT_ENVIRONMENT=.venv-spk uv sync --extra espnet-wavlm-joint
#   UV_PROJECT_ENVIRONMENT=.venv-spk HF_HOME=$WRKDIR/hf_cache \
#     uv run python -c "from espnet2.bin.spk_inference import Speech2Embedding; \
#       Speech2Embedding.from_pretrained(model_tag='espnet/voxcelebs12_ecapa_wavlm_joint')"
# NOTE: espnet-wavlm-joint conflicts with dac — that's why it lives in its OWN env (.venv-spk).

set -euo pipefail
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"
export UV_PROJECT_ENVIRONMENT=.venv-spk
export HF_HOME="${WRKDIR:-$HOME}/hf_cache"

uv run --offline -m quick_convert.cli.precompute \
    --config-name run/precompute_spk_wavlm_librispeech

#!/bin/bash
#SBATCH --job-name=dac_train
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu-a100-80g           
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/elec/t412-speechsynth/palp/vpc_collab/scripts/logs/pre_cont_%j.out
#SBATCH --error=/scratch/elec/t412-speechsynth/palp/vpc_collab/scripts/logs/pre_cont_%j.err

# ---------------------------------------------------------------------------
# GPU training of the controllable-RVQ anonymiser on precomputed DAC features.
# PREREQS (on the LOGIN node):
#   1. All features already precomputed (content/emo2vec/spk) + manifest + tokenizer.
#   2. Create the training env once:
#        UV_PROJECT_ENVIRONMENT=.venv-train uv sync --extra module-training --extra chatterbox
#      (add more extras if instantiation reports a missing SSL import)
# Then: sbatch slurm/train_dac.slurm
# ---------------------------------------------------------------------------

# GPU precompute of frozen DAC-16k content features.
# PREREQS (run on the LOGIN node, where there is internet):
#   1. Create the isolated env once:
#        UV_PROJECT_ENVIRONMENT=.venv-dac uv sync --extra dac
#   2. Warm the model cache once (compute nodes usually have no internet):
#        UV_PROJECT_ENVIRONMENT=.venv-dac HF_HOME=$WRKDIR/hf_cache \
#          uv run python -c "import dac; dac.DAC.load(dac.utils.download(model_type='16khz'))"
# Then: sbatch slurm/precompute_content_dac.slurm
# ---------------------------------------------------------------------------
 
set -euo pipefail
mkdir -p logs
cd "$SLURM_SUBMIT_DIR"
 
# isolated, pre-synced env so concurrent jobs don't clobber the shared .venv
export UV_PROJECT_ENVIRONMENT=.venv-dac
# persistent, pre-warmed model cache (no download needed at runtime)
export HF_HOME="${WRKDIR:-$HOME}/hf_cache"
 
# --offline: never hit the network from the compute node (fail loudly if cache missing)
uv run --offline -m quick_convert.cli.precompute \
    --config-name run/precompute_content_dac_librispeech

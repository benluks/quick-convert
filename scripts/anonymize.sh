#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run.sh <gpu_id> <config_path> [extra args...]

GPU_ID=${1:?Missing GPU id (e.g. 0)}
CONFIG=${2:?Missing config path (e.g. config.yaml)}
shift 2

# Set GPU
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Run python script with config as positional arg
uv run python -m scripts.anonymize "$CONFIG" "$@"
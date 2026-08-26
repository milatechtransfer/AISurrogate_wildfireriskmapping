#!/bin/bash
#SBATCH --job-name=model_smoke
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source .venv/bin/activate

: "${CONFIG:?CONFIG must point to a training YAML}"
smoke_args=(--config="$CONFIG")
if [[ -n "${DATA_ROOT:-}" ]]; then
    smoke_args+=(--data-root="$DATA_ROOT")
fi
python -m src.smoke_train_batch "${smoke_args[@]}"

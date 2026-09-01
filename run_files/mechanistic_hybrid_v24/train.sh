#!/bin/bash
#SBATCH --job-name=hybrid_v24_train
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --time=2-12:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:a100:1

set -euo pipefail

repo_root="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$repo_root"
mkdir -p logs
source .venv/bin/activate
export PYTHONUNBUFFERED=1

config="${CONFIG:-configs/mechanistic_hybrid_v24_reference.yaml}"
train_args=(--config="$config")
if [[ "${SKIP_FINAL_HEXEL_ARTIFACTS:-1}" == "1" ]]; then
    train_args+=(--no_log_test_predicted_hexels --no_log_val_predicted_hexels)
fi
python -m src.train "${train_args[@]}"

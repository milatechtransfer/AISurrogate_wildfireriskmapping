#!/bin/bash
#SBATCH --job-name=hybrid_v24_smoke
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:a100:1

set -euo pipefail

repo_root="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$repo_root"
mkdir -p logs
source .venv/bin/activate

config="${CONFIG:-configs/mechanistic_hybrid_v24_reference.yaml}"
python -m src.smoke_train_batch --config="$config"

#!/bin/bash
#SBATCH --job-name=mech_smoke
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
python -m src.smoke_train_batch \
    --config="configs/mechanistic/mechanistic_propagation_512_crop_128_firesize_quantiles.yaml"

#!/bin/bash
#SBATCH --job-name=mech_prop_512c128
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --time=15:59:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --signal=B:TERM@300

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

if [[ -z "${COMET_API_KEY:-}" ]]; then
    echo "ERROR: COMET_API_KEY must be exported." >&2
    exit 1
fi

python -m src.train \
    --config="configs/mechanistic/mechanistic_propagation_512_crop_128_firesize_quantiles.yaml"

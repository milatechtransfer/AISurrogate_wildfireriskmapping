#!/bin/bash
#SBATCH --job-name=cf_mean_weather_eval
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=unkillable
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16Gb
#SBATCH --gres=gpu:1

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
source .venv/bin/activate

python -m src.evaluate_counterfactual \
    --config configs/counterfactual_mean_weather_multi_output.yaml \
    --endpoint bp \
    --endpoint fi \
    --endpoint ros \
    --overwrite

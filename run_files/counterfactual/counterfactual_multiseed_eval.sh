#!/bin/bash
#SBATCH --job-name=cf_multiseed_eval
#SBATCH --output=logs/job_%x_%A_%a.out
#SBATCH --error=logs/job_%x_%A_%a.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16Gb
#SBATCH --gres=gpu:1
#SBATCH --exclude=cn-c034

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
source .venv/bin/activate

: "${CONFIG:?Submit with --export=ALL,CONFIG=<counterfactual-config>}"
: "${SLURM_ARRAY_TASK_ID:?Submit as an array job, e.g. --array=0-2}"

python -m src.evaluate_counterfactual \
    --config "${CONFIG}" \
    --run_id "${SLURM_ARRAY_TASK_ID}" \
    --overwrite

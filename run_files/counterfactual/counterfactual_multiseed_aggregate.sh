#!/bin/bash
#SBATCH --job-name=cf_multiseed_aggregate
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=long-cpu
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24Gb

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
source .venv/bin/activate

: "${CONFIG:?Submit with --export=ALL,CONFIG=<counterfactual-config>}"
: "${SCENARIO:?Submit with --export=ALL,SCENARIO=<scenario-name>}"

python -m src.aggregate_counterfactual_multirun_results \
    --config "${CONFIG}" \
    --scenario "${SCENARIO}"

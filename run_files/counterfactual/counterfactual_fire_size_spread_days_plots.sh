#!/bin/bash
#SBATCH --job-name=cf_fire_size_plots
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

config="configs/counterfactual/counterfactual_fire_size_spread_days_multi_output.yaml"
scenario="spread_days_q50_plus_0p7_q90_plus_5_beta2"
hex_id="16"
endpoints=("bp" "fi" "ros")

for endpoint in "${endpoints[@]}"; do
    python -m src.datasets.postprocessing.counterfactual.plotting.counterfactual_response_maps \
        --config "${config}" --scenario "${scenario}" --endpoint "${endpoint}" --hex_id "${hex_id}"
done

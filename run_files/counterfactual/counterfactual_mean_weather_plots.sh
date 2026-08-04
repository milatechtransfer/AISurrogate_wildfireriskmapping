#!/bin/bash
#SBATCH --job-name=cf_mean_weather_plots
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

config="configs/counterfactual_mean_weather_multi_output.yaml"
hex_id="16"
scenarios=(
    "bc_mean_weather_transplant"
)
endpoints=("bp" "fi" "ros")

for scenario in "${scenarios[@]}"; do
    for endpoint in "${endpoints[@]}"; do
        python -m src.datasets.postprocessing.counterfactual.plotting.counterfactual_response_maps \
            --config "${config}" --scenario "${scenario}" --endpoint "${endpoint}" --hex_id "${hex_id}"
    done
done

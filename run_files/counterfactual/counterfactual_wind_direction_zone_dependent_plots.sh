#!/bin/bash
#SBATCH --job-name=cf_wind_direction_zone_dependent_plots
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

config="configs/counterfactual/counterfactual_wind_direction_zone_dependent_multi_output.yaml"
hex_id="16"
scenarios=(
    "wind_dir_000" "wind_dir_045" "wind_dir_090" "wind_dir_135" "wind_dir_180"
    "wind_dir_225" "wind_dir_270" "wind_dir_315" "wind_dir_360"
)
endpoints=("bp" "fi" "ros")

for scenario in "${scenarios[@]}"; do
    for endpoint in "${endpoints[@]}"; do
        python -m src.datasets.postprocessing.counterfactual.plotting.counterfactual_response_maps \
            --config "${config}" --scenario "${scenario}" --endpoint "${endpoint}" --hex_id "${hex_id}" --zone_overlay
    done
done

# Compass roses: 8 direction-diff maps arranged around a circle (0deg = N, clockwise), one per endpoint.
for endpoint in "${endpoints[@]}"; do
    python -m src.datasets.postprocessing.counterfactual.plotting.counterfactual_wind_direction_compass \
        --config "${config}" --endpoint "${endpoint}" --hex_id "${hex_id}" --zone_overlay
done

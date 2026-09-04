#!/bin/bash
#SBATCH --job-name=cf_fuel_polygons_plots
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=long-cpu
#SBATCH --ntasks=1
#SBATCH --mem=16Gb
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
source .venv/bin/activate

config="configs/counterfactual/counterfactual_fuel_polygons_multi_output.yaml"
hex_id="16"
scenarios=(
    "pooled_burn_scars_to_aspen"
)
endpoints=("bp" "fi" "ros")

for scenario in "${scenarios[@]}"; do
    python -m src.datasets.postprocessing.counterfactual.plotting.counterfactual_fuel_intervention_map \
        --config "${config}" --scenario "${scenario}" --endpoint bp --hex_id "${hex_id}"
    python -m src.datasets.postprocessing.counterfactual.plotting.counterfactual_local_zoom_panels \
        --config "${config}" --scenario "${scenario}" --hex_id "${hex_id}"

    for endpoint in "${endpoints[@]}"; do
        python -m src.datasets.postprocessing.counterfactual.plotting.counterfactual_response_maps \
            --config "${config}" --scenario "${scenario}" --endpoint "${endpoint}" --hex_id "${hex_id}"
        python -m src.datasets.postprocessing.counterfactual.plotting.counterfactual_change_distribution \
            --config "${config}" --scenario "${scenario}" --endpoint "${endpoint}" --hex_id "${hex_id}"
    done
done

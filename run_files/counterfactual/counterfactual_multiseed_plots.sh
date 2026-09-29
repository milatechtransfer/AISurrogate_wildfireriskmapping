#!/bin/bash
#SBATCH --job-name=cf_multiseed_plots
#SBATCH --output=logs/job_%x_%A_%a.out
#SBATCH --error=logs/job_%x_%A_%a.err
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
: "${SCENARIO_KIND:?Submit with --export=ALL,SCENARIO_KIND=<fuel|weather|fire_size>}"
: "${SLURM_ARRAY_TASK_ID:?Submit as an array job, e.g. --array=0-2}"

case "${SCENARIO_KIND}" in
    fuel|weather|fire_size) ;;
    *) echo "Unsupported SCENARIO_KIND=${SCENARIO_KIND}" >&2; exit 2 ;;
esac

mapfile -t run_info < <(
    python -c '
import sys
from pathlib import Path
from src.config import SEEDS
from src.datasets.postprocessing.counterfactual.counterfactual_base import load_counterfactual_config, resolve_project_path

config = load_counterfactual_config(Path(sys.argv[1]))
seed = SEEDS[int(sys.argv[2])]
print(seed)
print(resolve_project_path(config.save_dir) / f"seed_{seed}")
print(*config.hex_ids, sep="\n")
' "${CONFIG}" "${SLURM_ARRAY_TASK_ID}"
)

seed="${run_info[0]}"
experiment_dir="${run_info[1]}"
hex_ids=("${run_info[@]:2}")
endpoints=("bp" "fi" "ros")

for hex_id in "${hex_ids[@]}"; do
    if [[ "${SCENARIO_KIND}" == "fuel" ]]; then
        python -m src.datasets.postprocessing.counterfactual.plotting.counterfactual_fuel_intervention_map \
            --config "${CONFIG}" \
            --experiment_dir "${experiment_dir}" \
            --scenario "${SCENARIO}" \
            --endpoint bp \
            --hex_id "${hex_id}"
    fi

    for endpoint in "${endpoints[@]}"; do
        python -m src.datasets.postprocessing.counterfactual.plotting.counterfactual_response_maps \
            --config "${CONFIG}" \
            --experiment_dir "${experiment_dir}" \
            --scenario "${SCENARIO}" \
            --endpoint "${endpoint}" \
            --hex_id "${hex_id}"
        if [[ "${SCENARIO_KIND}" == "fuel" ]]; then
            python -m src.datasets.postprocessing.counterfactual.plotting.counterfactual_change_distribution \
                --config "${CONFIG}" \
                --experiment_dir "${experiment_dir}" \
                --scenario "${SCENARIO}" \
                --endpoint "${endpoint}" \
                --hex_id "${hex_id}"
        fi
    done
done

echo "Completed seed ${seed} plots under ${experiment_dir}."

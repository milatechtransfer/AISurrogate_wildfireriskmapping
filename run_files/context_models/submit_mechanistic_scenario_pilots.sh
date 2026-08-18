#!/bin/bash

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

script="run_files/context_models/train_model.sh"

submit() {
    local job_name=$1
    local config=$2
    local job_id
    job_id=$(sbatch --parsable --job-name="$job_name" --export="ALL,CONFIG=$config" "$script")
    printf '%s %s %s\n' "$job_id" "$job_name" "$config"
}

submit "mechv21_pilot" "configs/mechanistic/mechanistic_propagation_v21_512_crop_256_firesize_q3_pilot.yaml"
submit "mechv3_pilot" "configs/mechanistic/mechanistic_propagation_v3_512_crop_256_spread_opportunity_q3_pilot.yaml"

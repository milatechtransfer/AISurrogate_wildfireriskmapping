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

submit "ctx512_nofs" "configs/context_models/unet_512_crop_256_no_firesize.yaml"
submit "ctx512_q3" "configs/context_models/unet_512_crop_256_firesize_q3.yaml"
submit "mechv2_512q3" "configs/mechanistic/mechanistic_propagation_v2_512_crop_256_firesize_q3.yaml"
submit "unetphys_512q3" "configs/mechanistic/unet_512_crop_256_firesize_q3_physical_control.yaml"

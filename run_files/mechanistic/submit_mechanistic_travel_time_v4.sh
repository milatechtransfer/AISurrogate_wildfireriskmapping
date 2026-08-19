#!/bin/bash

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

config="configs/mechanistic/mechanistic_travel_time_v4_512_crop_256_spread_opportunity_q3.yaml"
script="run_files/context_models/train_model.sh"
dependency_job_id="${DEPENDENCY_JOB_ID:-}"
scenario_checkpoint="${SCENARIO_UNET_CHECKPOINT:?SCENARIO_UNET_CHECKPOINT must point to the trained scenario U-Net best.pth}"
sbatch_args=(
    --parsable
    --job-name="mech_travel_v4"
    --export="ALL,CONFIG=$config,SCENARIO_UNET_CHECKPOINT=$scenario_checkpoint"
)
if [[ -n "$dependency_job_id" ]]; then
    sbatch_args+=(--dependency="afterok:$dependency_job_id")
fi

job_id=$(sbatch "${sbatch_args[@]}" "$script")
printf '%s %s %s dependency=%s\n' "$job_id" "mech_travel_v4" "$config" "${dependency_job_id:-none}"

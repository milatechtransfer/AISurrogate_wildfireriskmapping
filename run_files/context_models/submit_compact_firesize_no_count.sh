#!/bin/bash

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

train_script="run_files/context_models/train_model.sh"
compact_256_config="configs/context_models/unet_compact_256_firesize_q3.yaml"
compact_512_config="configs/context_models/unet_compact_512_crop_256_firesize_q3.yaml"
mechanistic_config="configs/mechanistic/mechanistic_travel_time_compact_512_crop_256_firesize_q3.yaml"
compact_512_checkpoint="experiments/context_models/unet_compact_512_crop_256_firesize_q3/best.pth"

compact_256_job_id=$(sbatch \
    --parsable \
    --job-name="ctx256_q3_compact" \
    --export="ALL,CONFIG=$compact_256_config" \
    "$train_script")

compact_512_job_id=$(sbatch \
    --parsable \
    --job-name="ctx512_q3_compact" \
    --export="ALL,CONFIG=$compact_512_config" \
    "$train_script")

mechanistic_job_id=$(sbatch \
    --parsable \
    --job-name="mech_q3_compact" \
    --dependency="afterok:$compact_512_job_id" \
    --export="ALL,CONFIG=$mechanistic_config,SCENARIO_UNET_CHECKPOINT=$compact_512_checkpoint" \
    "$train_script")

printf '%s %s\n' "$compact_256_job_id" "compact_unet_256_firesize_q3"
printf '%s %s\n' "$compact_512_job_id" "compact_unet_512_firesize_q3"
printf '%s %s dependency=%s\n' "$mechanistic_job_id" "compact_mechanistic_firesize_q3_no_count" "$compact_512_job_id"

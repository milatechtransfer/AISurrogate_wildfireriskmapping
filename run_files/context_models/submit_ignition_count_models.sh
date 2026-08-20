#!/bin/bash

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

train_script="run_files/context_models/train_model.sh"
base_256_data_root="/network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA/data_samples_v4"
unet_256_baseline_config="configs/context_models/unet_256_firesize_q3.yaml"
unet_256_config="configs/context_models/unet_256_firesize_q3_ignition_count.yaml"
full_config="configs/context_models/unet_512_crop_256_firesize_q3_ignition_count.yaml"
compact_config="configs/context_models/unet_compact_512_crop_256_firesize_q3_ignition_count.yaml"
mechanistic_config="configs/mechanistic/mechanistic_travel_time_compact_512_crop_256_firesize_q3_ignition_count.yaml"
compact_checkpoint="experiments/context_models/unet_compact_512_crop_256_firesize_q3_ignition_count/best.pth"

prepare_512_job_id=$(sbatch \
    --parsable \
    "run_files/context_models/prepare_ignition_count.sh")

prepare_256_job_id=$(sbatch \
    --parsable \
    --export="ALL,DATA_ROOT=$base_256_data_root" \
    "run_files/context_models/prepare_ignition_count.sh")

unet_256_baseline_job_id=$(sbatch \
    --parsable \
    --job-name="ctx256_q3" \
    --export="ALL,CONFIG=$unet_256_baseline_config" \
    "$train_script")

unet_256_job_id=$(sbatch \
    --parsable \
    --job-name="ctx256_count_full" \
    --dependency="afterok:$prepare_256_job_id" \
    --export="ALL,CONFIG=$unet_256_config" \
    "$train_script")

full_job_id=$(sbatch \
    --parsable \
    --job-name="ctx512_count_full" \
    --dependency="afterok:$prepare_512_job_id" \
    --export="ALL,CONFIG=$full_config" \
    "$train_script")

data_job_id=$(sbatch \
    --parsable \
    --dependency="afterok:$prepare_512_job_id" \
    "run_files/context_models/generate_probability_mass_context.sh")

compact_job_id=$(sbatch \
    --parsable \
    --job-name="ctx512_count_compact" \
    --dependency="afterok:$data_job_id" \
    --export="ALL,CONFIG=$compact_config" \
    "$train_script")

mechanistic_job_id=$(sbatch \
    --parsable \
    --job-name="mech_count_compact" \
    --dependency="afterok:$compact_job_id" \
    --export="ALL,CONFIG=$mechanistic_config,SCENARIO_UNET_CHECKPOINT=$compact_checkpoint" \
    "$train_script")

printf '%s %s\n' "$prepare_512_job_id" "prepare_ignition_count_512"
printf '%s %s\n' "$prepare_256_job_id" "prepare_ignition_count_256"
printf '%s %s\n' "$unet_256_baseline_job_id" "full_unet_256_no_count"
printf '%s %s dependency=%s\n' "$unet_256_job_id" "full_count_unet_256" "$prepare_256_job_id"
printf '%s %s dependency=%s\n' "$full_job_id" "full_count_unet_512" "$prepare_512_job_id"
printf '%s %s dependency=%s\n' "$data_job_id" "probability_mass_context_array" "$prepare_512_job_id"
printf '%s %s dependency=%s\n' "$compact_job_id" "compact_count_unet" "$data_job_id"
printf '%s %s dependency=%s\n' "$mechanistic_job_id" "compact_count_mechanistic" "$compact_job_id"

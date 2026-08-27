#!/bin/bash

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

source_data_root="${SOURCE_DATA_ROOT:-/network/scratch/o/olutayot/nrcan_wildfireriskmapping/data_samples/data_samples_v6_native_context_512_crop_256_ignition_probability_mass}"
data_root="${DATA_ROOT:-/network/scratch/o/olutayot/nrcan_wildfireriskmapping/data_samples/data_samples_v6_native_context_512_crop_256_ignition_probability_mass_log_firesize}"
config="${CONFIG:-configs/mechanistic/mechanistic_hybrid_v23_native_512_crop_256_firesize_q3.yaml}"

prepare_job_id=$(sbatch \
    --parsable \
    --export="ALL,SOURCE_DATA_ROOT=$source_data_root,DATA_ROOT=$data_root" \
    run_files/mechanistic/prepare_hybrid_v23.sh)

smoke_job_id=$(sbatch \
    --parsable \
    --dependency="afterok:$prepare_job_id" \
    --constraint=ampere \
    --gres=gpu:a100:1 \
    --export="ALL,CONFIG=$config,DATA_ROOT=$data_root" \
    run_files/context_models/smoke_model.sh)

train_job_id=$(sbatch \
    --parsable \
    --dependency="afterok:$smoke_job_id" \
    --constraint=ampere \
    --gres=gpu:a100:1 \
    --time=2-12:00:00 \
    --job-name=hybrid_v23_native \
    --export="ALL,CONFIG=$config,DATA_ROOT=$data_root,SKIP_FINAL_HEXEL_ARTIFACTS=1" \
    run_files/context_models/train_model.sh)

printf '%s %s source=%s data=%s\n' "$prepare_job_id" "v23_data" "$source_data_root" "$data_root"
printf '%s %s dependency=%s gres=gpu:a100:1\n' "$smoke_job_id" "hybrid_v23_gpu_smoke" "$prepare_job_id"
printf '%s %s dependency=%s gres=gpu:a100:1\n' "$train_job_id" "hybrid_v23_train" "$smoke_job_id"

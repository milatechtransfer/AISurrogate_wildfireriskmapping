#!/bin/bash

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

data_root="${DATA_ROOT:-/network/scratch/o/olutayot/nrcan_wildfireriskmapping/data_samples/data_samples_v6_native_context_512_crop_256_ignition_probability_mass}"
config="${CONFIG:-configs/mechanistic/mechanistic_hybrid_v22_native_512_crop_256_firesize_q3.yaml}"

prepare_job_id=$(sbatch \
    --parsable \
    --export="ALL,DATA_ROOT=$data_root" \
    run_files/mechanistic/prepare_hybrid_v22_native.sh)

finalize_job_id=$(sbatch \
    --parsable \
    --dependency="afterok:$prepare_job_id" \
    --export="ALL,DATA_ROOT=$data_root" \
    run_files/mechanistic/finalize_hybrid_v22_native.sh)

smoke_job_id=$(sbatch \
    --parsable \
    --dependency="afterok:$finalize_job_id" \
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
    --job-name=hybrid_v22_native \
    --export="ALL,CONFIG=$config,DATA_ROOT=$data_root,SKIP_FINAL_HEXEL_ARTIFACTS=1" \
    run_files/context_models/train_model.sh)

printf '%s %s data=%s\n' "$prepare_job_id" "native_data_array" "$data_root"
printf '%s %s dependency=%s\n' "$finalize_job_id" "native_data_finalize" "$prepare_job_id"
printf '%s %s dependency=%s gres=gpu:a100:1\n' "$smoke_job_id" "hybrid_v22_gpu_smoke" "$finalize_job_id"
printf '%s %s dependency=%s gres=gpu:a100:1\n' "$train_job_id" "hybrid_v22_train" "$smoke_job_id"

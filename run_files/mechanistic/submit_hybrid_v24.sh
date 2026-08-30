#!/bin/bash

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

data_root="${DATA_ROOT:-/network/scratch/o/olutayot/nrcan_wildfireriskmapping/data_samples/data_samples_v6_native_context_512_crop_256_ignition_probability_mass_log_firesize}"
config="${CONFIG:-configs/mechanistic/mechanistic_hybrid_v24_native_512_crop_256_firesize_q3.yaml}"

if [[ ! -f "$data_root/fire_size_log_stats.json" ]]; then
    echo "ERROR: prepared v2.3/v2.4 data root is missing: $data_root" >&2
    exit 1
fi

smoke_job_id=$(sbatch \
    --parsable \
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
    --job-name=hybrid_v24_native \
    --export="ALL,CONFIG=$config,DATA_ROOT=$data_root,SKIP_FINAL_HEXEL_ARTIFACTS=1" \
    run_files/context_models/train_model.sh)

printf '%s %s data=%s\n' "$smoke_job_id" "hybrid_v24_gpu_smoke" "$data_root"
printf '%s %s dependency=%s gres=gpu:a100:1\n' "$train_job_id" "hybrid_v24_train" "$smoke_job_id"

#!/bin/bash

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

config="configs/mechanistic/interpretable_physics_v2_count_sources_512_crop_256_firesize_q3.yaml"
script="run_files/context_models/train_model.sh"

job_id=$(sbatch \
    --parsable \
    --constraint="ampere" \
    --gres="gpu:a100:1" \
    --job-name="interp_physics_v2" \
    --export="ALL,CONFIG=$config,SKIP_FINAL_HEXEL_ARTIFACTS=1" \
    "$script")

printf '%s %s %s dependency=none gres=gpu:a100:1\n' "$job_id" "interp_physics_v2" "$config"

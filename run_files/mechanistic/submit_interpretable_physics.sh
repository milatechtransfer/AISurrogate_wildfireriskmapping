#!/bin/bash

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

config="configs/mechanistic/interpretable_physics_512_crop_256_firesize_q3.yaml"
script="run_files/context_models/train_model.sh"

job_id=$(sbatch \
    --parsable \
    --job-name="interp_physics" \
    --export="ALL,CONFIG=$config" \
    "$script")

printf '%s %s %s dependency=none\n' "$job_id" "interp_physics" "$config"

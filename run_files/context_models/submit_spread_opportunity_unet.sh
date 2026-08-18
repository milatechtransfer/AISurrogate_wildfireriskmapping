#!/bin/bash

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

config="configs/context_models/unet_512_crop_256_spread_opportunity_q3.yaml"
script="run_files/context_models/train_model.sh"
job_id=$(sbatch --parsable --job-name="ctx512_spreadq3" --export="ALL,CONFIG=$config" "$script")
printf '%s %s %s\n' "$job_id" "ctx512_spreadq3" "$config"

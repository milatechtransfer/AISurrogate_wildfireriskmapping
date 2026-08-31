#!/bin/bash

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

source_data_root="${SOURCE_DATA_ROOT:-/network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA/data_samples_v4}"
config="${CONFIG:-configs/context_models/unet_256_firesize_q3_raw_hectares_zscore.yaml}"
data_root=$(
    .venv/bin/python - "$config" <<'PY'
import sys
from pathlib import Path

import yaml

with Path(sys.argv[1]).open() as handle:
    config = yaml.safe_load(handle)

print(Path(config["data"]["root_dir"]).expanduser().resolve())
PY
)

prepare_job_id=$(sbatch \
    --parsable \
    --export="ALL,SOURCE_DATA_ROOT=$source_data_root,DATA_ROOT=$data_root" \
    run_files/context_models/prepare_raw_hectares_zscore_256.sh)

train_job_id=$(sbatch \
    --parsable \
    --dependency="afterok:$prepare_job_id" \
    --constraint=ampere \
    --gres=gpu:a100:1 \
    --time=2-12:00:00 \
    --job-name=unet256_raw_ha_z \
    run_files/train_no_tmp_copy.sh \
    "$config")

printf '%s %s source=%s data=%s\n' "$prepare_job_id" "raw_hectares_zscore_data" "$source_data_root" "$data_root"
printf '%s %s dependency=%s gres=gpu:a100:1\n' "$train_job_id" "raw_hectares_zscore_train" "$prepare_job_id"

#!/bin/bash
#SBATCH --job-name=v22_native_finalize
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source .venv/bin/activate
mkdir -p logs

raw_root="/network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA"
data_root="${DATA_ROOT:-/network/scratch/o/olutayot/nrcan_wildfireriskmapping/data_samples/data_samples_v6_native_context_512_crop_256_ignition_probability_mass}"
curve_source="/network/scratch/o/olutayot/nrcan_wildfireriskmapping/data_samples/data_samples_v5_context_512_crop_256_ignition_probability_mass/fbp_curves_national_fuel.csv"

python -m data_preparation.split_data \
    --data_dir="$data_root" \
    --val_hex_id 02 18 23 33 46 \
    --test_hex_id 01 12 16 39 49

cp -f "$curve_source" "$data_root/fbp_curves_national_fuel.csv"

python -m data_preparation.process_tabular_data \
    --root_dir="$raw_root" \
    --save_dir="$data_root" \
    --train_split_file=train_indices.csv \
    --modelling_approach=1 \
    --weather_output_file=weather_table_processed.csv \
    --fire_size_input_file=df_fire_fru.csv \
    --fire_size_output_file=df_fire_fru_processed.csv \
    --weather_norm_params_file=weather_norm_params.json \
    --fire_size_norm_params_file=fire_size_norm_params.json

python -m data_preparation.process_ignition_count \
    --raw_root="$raw_root" \
    --output_path="$data_root/ignition_count_processed.csv" \
    --train_split_path="$data_root/train_indices.csv" \
    --norm_params_path="$data_root/ignition_count_norm_params.json" \
    --mask_scope=actual

python -m data_preparation.compute_dataset_normalization_stats \
    --raw_data_dir="$raw_root" \
    --root_dir="$data_root" \
    --save_dir="$data_root" \
    --types elevation fuel_curve_iROS fuel_curve_HFI fuel_curve_iROS_HFI fire_intensity fire_ros fire_burn_probability \
    --overwrite

python - "$data_root" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
expected = {
    "train": {3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 17, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 40, 41, 42, 43, 44, 45, 47, 48, 50, 51, 52, 53, 54},
    "val": {2, 18, 23, 33, 46},
    "test": {1, 12, 16, 39, 49},
}
for split, expected_ids in expected.items():
    frame = pd.read_csv(root / f"{split}_indices.csv")
    actual_ids = set(frame["hex_id"].astype(int).unique())
    if actual_ids != expected_ids:
        raise RuntimeError(f"{split} hex IDs differ: expected {sorted(expected_ids)}, got {sorted(actual_ids)}")
    missing = [name for name in frame["filename"] if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"{split} references {len(missing)} missing patches; first={missing[0]}")

feature_map = json.loads((root / "feature_channel_map_1.json").read_text())
expected_features = {
    "fuel_grid",
    "elevation_grid",
    "ignition_grid_human",
    "ignition_grid_lightning",
    "firezones_grid",
    "bp_out_grid",
    "fi_out_grid",
    "ros_out_grid",
}
if set(feature_map) != expected_features:
    raise RuntimeError(f"Unexpected feature map keys: {sorted(feature_map)}")
print("Native v2.2 dataset finalized and verified.")
PY

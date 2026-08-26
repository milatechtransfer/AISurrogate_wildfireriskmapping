#!/bin/bash
#SBATCH --job-name=v22_native_data
#SBATCH --output=logs/array_%x_%A_%a.out
#SBATCH --error=logs/array_%x_%A_%a.err
#SBATCH --partition=long
#SBATCH --array=0-53%12
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source .venv/bin/activate
mkdir -p logs

task_id=${SLURM_ARRAY_TASK_ID:-0}
num_tasks=$((${SLURM_ARRAY_TASK_MAX:-0} - ${SLURM_ARRAY_TASK_MIN:-0} + 1))
raw_root="/network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA"
data_root="${DATA_ROOT:-/network/scratch/o/olutayot/nrcan_wildfireriskmapping/data_samples/data_samples_v6_native_context_512_crop_256_ignition_probability_mass}"

python -m data_preparation.process_hexels_into_grids \
    --root_dir="$raw_root" \
    --save_dir="$data_root" \
    --modelling_approach=1 \
    --win_h=512 \
    --win_w=512 \
    --target_crop_h=256 \
    --target_crop_w=256 \
    --overlap_ratio=0.2 \
    --ignition_weighting=probability_mass \
    --fuel_grid_representation=raw \
    --mask_scope=actual \
    --preserve_native_grid \
    --overwrite \
    --is_array_job \
    --task_id="$task_id" \
    --num_tasks="$num_tasks"

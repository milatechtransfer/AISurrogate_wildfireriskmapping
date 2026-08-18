#!/bin/bash
#SBATCH --job-name=spread_opportunity
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source .venv/bin/activate
mkdir -p logs

raw_root="/network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA"
data_root="/network/scratch/o/olutayot/nrcan_wildfireriskmapping/data_samples/data_samples_v4_context_512_crop_256"

python -m data_preparation.process_spread_opportunity \
    --raw_root="$raw_root" \
    --output_path="$data_root/spread_opportunity_processed.csv" \
    --train_split_path="$data_root/train_indices.csv" \
    --norm_params_path="$data_root/spread_opportunity_norm_params.json" \
    --mask_scope=actual

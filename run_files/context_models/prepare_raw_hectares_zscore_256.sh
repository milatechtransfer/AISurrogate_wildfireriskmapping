#!/bin/bash
#SBATCH --job-name=raw_ha_z_data
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source .venv/bin/activate
mkdir -p logs

source_data_root="${SOURCE_DATA_ROOT:-/network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA/data_samples_v4}"
data_root="${DATA_ROOT:-/network/scratch/o/olutayot/nrcan_wildfireriskmapping/data_samples/data_samples_v4_raw_hectares_zscore}"

python -m data_preparation.prepare_raw_hectares_zscore_dataset \
    --source-root="$source_data_root" \
    --destination-root="$data_root"

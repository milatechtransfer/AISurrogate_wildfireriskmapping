#!/bin/bash
#SBATCH --job-name=v23_data
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

source_data_root="${SOURCE_DATA_ROOT:-/network/scratch/o/olutayot/nrcan_wildfireriskmapping/data_samples/data_samples_v6_native_context_512_crop_256_ignition_probability_mass}"
data_root="${DATA_ROOT:-/network/scratch/o/olutayot/nrcan_wildfireriskmapping/data_samples/data_samples_v6_native_context_512_crop_256_ignition_probability_mass_log_firesize}"

python -m data_preparation.prepare_hybrid_v23_dataset \
    --source-root="$source_data_root" \
    --destination-root="$data_root"

#!/bin/bash
#SBATCH --job-name=prep_512c128
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=long-cpu
#SBATCH --ntasks=1
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source .venv/bin/activate

python -m data_preparation.prepare_context_crop_view \
    --source_dir="/network/scratch/o/olutayot/nrcan_wildfireriskmapping/data_samples/data_samples_v4_context_512_crop_256" \
    --save_dir="/network/scratch/o/olutayot/nrcan_wildfireriskmapping/data_samples/data_samples_v4_context_512_center_128" \
    --target_crop_h=128 \
    --target_crop_w=128

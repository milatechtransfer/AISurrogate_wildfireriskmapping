#!/bin/bash
#SBATCH --job-name=grid_gen
#SBATCH --output=logs/array_%x_%A_%a.log
#SBATCH --array=0-53 # hard-coded since we know there's 54 hexel subdirs
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2

mkdir -p logs
source .venv/bin/activate

# Robust calculation of Num Tasks (Max Index - Min Index + 1)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
# Default to 1 if not running in array
if [ -n "$SLURM_ARRAY_TASK_MAX" ]; then
    NUM_TASKS=$((SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1))
else
    NUM_TASKS=1
fi

echo "Starting Worker $TASK_ID / $NUM_TASKS"

# If save_dir is kept as None, the grids will be saved to 'data_samples_modelling_approach_<VERSION_NUMBER_HERE>/' directory in the `root_dir/`
python -m data_preparation.process_hexels_into_grids \
    --root_dir="/network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA/" \
    --save_dir="/network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA/data_samples_v4" \
    --modelling_approach=1 \
    --win_h=256 \
    --win_w=256 \
    --overlap_ratio=0.2 \
    --ignition_weighting="distribution" \
    --fuel_grid_representation="raw" \
    --is_array_job \
    --task_id=$TASK_ID \
    --num_tasks=$NUM_TASKS

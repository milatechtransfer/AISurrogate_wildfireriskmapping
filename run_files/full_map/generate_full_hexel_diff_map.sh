#!/bin/bash
##SBATCH --mail-type=all
##SBATCH --mail-user=name@mila.quebec
#SBATCH --job-name=full_map_diff
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --mem=64Gb
#SBATCH --cpus-per-task=4

set -euo pipefail

# This step is CPU-only (no model/GPU involved) but still loads two full national-grid
# rasters (predicted mosaic + ground truth) into memory per target, sequentially.
# Adjust --mem above to comfortably fit 2x (height * width * 4 bytes) for your rasters.

# Capture the first argument, default to the common pipeline config.
CONFIG_FILE=${1:-configs/bp_common_input_pipeline.yaml}
MOSAIC_DIR=${MOSAIC_DIR:-experiments/full_map}
OUTPUT_DIR=${OUTPUT_DIR:-experiments/full_map}
# Example: DIFF_ARGS="--save-plots --title='Burn Probability'"
DIFF_ARGS=${DIFF_ARGS:-}

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
source .venv/bin/activate
# Flush stdout/stderr immediately so log lines aren't lost if the job is
# preempted before Python's internal buffers would otherwise flush.
export PYTHONUNBUFFERED=1

echo "Diffing national mosaics against ground truth with config: $CONFIG_FILE"
read -r -a DIFF_ARG_ARRAY <<< "$DIFF_ARGS"
python -m src.full_map.generate_full_hexel_diff_map \
    --config="$CONFIG_FILE" \
    --mosaic-dir="$MOSAIC_DIR" \
    --output-dir="$OUTPUT_DIR" \
    "${DIFF_ARG_ARRAY[@]}"

#!/bin/bash
##SBATCH --mail-type=all
##SBATCH --mail-user=name@mila.quebec
#SBATCH --job-name=full_map_hazard
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mem=64Gb
#SBATCH --cpus-per-task=4

set -euo pipefail

# CPU-only, memory-heavy: ground truth and prediction are processed sequentially (never both
# loaded at once), and bp/fi arrays are combined in place, but each phase still briefly holds
# 2 full national-grid float32 arrays (bp + fi) plus one int32 output array. For a Canada-wide
# raster at 100m resolution (~55,000 x 46,000 px, ~10GB per float32 array, ~5GB per int32
# array) that's ~25GB peak -- 64Gb gives comfortable headroom. Adjust --mem to fit your
# reference raster's actual (height * width * 4 bytes) footprint. Check it with:
#   python -c "import rasterio; s = rasterio.open('<path>'); print(s.height, s.width, s.height*s.width*4/1e9, 'GB')"

# Capture the first argument, default to the common pipeline config.
CONFIG_FILE=${1:-configs/bp_common_input_pipeline.yaml}
# Directory containing bp/fi_national_predicted_map.tif (from generate_full_hexel_map.py).
# Not required if GT_ONLY=1.
MOSAIC_DIR=${MOSAIC_DIR:-experiments/full_map}
OUTPUT_DIR=${OUTPUT_DIR:-experiments/full_map}
# Set to 1 to only compute the ground-truth hazard map (no predicted mosaics needed).
GT_ONLY=${GT_ONLY:-0}
# Example: HAZARD_ARGS="--save-plots --title='National Hazard'"
# Resuming is automatic: existing hazard_national_*_map.tif files are skipped by default. To
# force a full recompute instead, use:
#   HAZARD_ARGS="--force-recompute"
HAZARD_ARGS=${HAZARD_ARGS:-}

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
source .venv/bin/activate
# Flush stdout/stderr immediately so log lines aren't lost if the job is
# preempted before Python's internal buffers would otherwise flush.
export PYTHONUNBUFFERED=1

echo "Computing national hazard map(s) with config: $CONFIG_FILE"
read -r -a HAZARD_ARG_ARRAY <<< "$HAZARD_ARGS"
GT_ONLY_ARGS=()
MOSAIC_DIR_ARGS=()
if [[ "$GT_ONLY" == "1" ]]; then
    GT_ONLY_ARGS=(--gt-only)
else
    MOSAIC_DIR_ARGS=(--mosaic-dir="$MOSAIC_DIR")
fi
python -m src.full_map.generate_national_hazard_map \
    --config="$CONFIG_FILE" \
    --output-dir="$OUTPUT_DIR" \
    "${MOSAIC_DIR_ARGS[@]}" \
    "${GT_ONLY_ARGS[@]}" \
    "${HAZARD_ARG_ARRAY[@]}"

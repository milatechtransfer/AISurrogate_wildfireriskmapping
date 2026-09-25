#!/bin/bash
##SBATCH --mail-type=all
##SBATCH --mail-user=name@mila.quebec
#SBATCH --job-name=full_map_mosaic
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mem=48Gb
#SBATCH --cpus-per-task=4

set -euo pipefail

# This step is CPU-only (no model/GPU involved) but can be memory-heavy: it holds one
# full national-grid float32 array in memory per target (built incrementally via small
# per-hexel windows, so no extra full-size copies beyond that one array). For a
# Canada-wide raster at 100m resolution (~55,000 x 46,000 px) that's ~10GB -- 48Gb gives
# comfortable headroom. Adjust --mem to fit your reference raster's actual
# (height * width * 4 bytes) footprint. Check it with:
#   python -c "import rasterio; s = rasterio.open('<path>'); print(s.height, s.width, s.height*s.width*4/1e9, 'GB')"

# Capture the first argument, default to the common pipeline config.
CONFIG_FILE=${1:-configs/bp_common_input_pipeline.yaml}
# Directory containing per-split predicted hexels (defaults to config.save_dir if unset).
PRED_ROOT=${PRED_ROOT:-}
OUTPUT_DIR=${OUTPUT_DIR:-experiments/full_map}
# Set GT_MODE=1 to mosaic ground-truth hexels (stitched from raw per-hexel rasters) instead of
# predicted hexels, writing {target}_national_gt_map.tif instead of
# {target}_national_predicted_map.tif. Only used with GT_MODE=1: RAW_DATA_DIR (defaults to
# config.data.raw_data_dir if unset) and BACKFILL_FROM_REFERENCE=1 (fills any pixel still
# nodata after per-hexel stitching from the seamless, already-merged
# config.full_map.national_gt_raster_paths raster).
GT_MODE=${GT_MODE:-0}
RAW_DATA_DIR=${RAW_DATA_DIR:-}
BACKFILL_FROM_REFERENCE=${BACKFILL_FROM_REFERENCE:-0}
# Example: MOSAIC_ARGS="--save-plots --scale=log --title='Burn Probability'"
# Resuming is automatic: existing output mosaic files are skipped by default. To force a full
# recompute instead, use:
#   MOSAIC_ARGS="--force-recompute"
MOSAIC_ARGS=${MOSAIC_ARGS:-}

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
source .venv/bin/activate
# Flush stdout/stderr immediately so log lines aren't lost if the job is
# preempted before Python's internal buffers would otherwise flush.
export PYTHONUNBUFFERED=1

read -r -a MOSAIC_ARG_ARRAY <<< "$MOSAIC_ARGS"
EXTRA_ARGS=()
if [[ "$GT_MODE" == "1" ]]; then
    echo "Mosaicking ground-truth hexels onto the national grid with config: $CONFIG_FILE"
    EXTRA_ARGS+=(--gt)
    if [[ -n "$RAW_DATA_DIR" ]]; then
        EXTRA_ARGS+=(--raw-data-dir="$RAW_DATA_DIR")
    fi
    if [[ "$BACKFILL_FROM_REFERENCE" == "1" ]]; then
        EXTRA_ARGS+=(--backfill-from-reference)
    fi
else
    echo "Mosaicking predicted hexels onto the national grid with config: $CONFIG_FILE"
    if [[ -n "$PRED_ROOT" ]]; then
        EXTRA_ARGS+=(--pred-root="$PRED_ROOT")
    fi
fi
python -m src.full_map.generate_full_hexel_map \
    --config="$CONFIG_FILE" \
    --output-dir="$OUTPUT_DIR" \
    "${EXTRA_ARGS[@]}" \
    "${MOSAIC_ARG_ARRAY[@]}"

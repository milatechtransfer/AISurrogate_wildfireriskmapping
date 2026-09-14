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
# Example: MOSAIC_ARGS="--save-plots --scale=log --title='Burn Probability'"
# Resuming is automatic: existing {target}_national_predicted_map.tif files are skipped by
# default. To force a full recompute instead, use:
#   MOSAIC_ARGS="--force-recompute"
MOSAIC_ARGS=${MOSAIC_ARGS:-}

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
source .venv/bin/activate
# Flush stdout/stderr immediately so log lines aren't lost if the job is
# preempted before Python's internal buffers would otherwise flush.
export PYTHONUNBUFFERED=1

echo "Mosaicking predicted hexels onto the national grid with config: $CONFIG_FILE"
read -r -a MOSAIC_ARG_ARRAY <<< "$MOSAIC_ARGS"
PRED_ROOT_ARGS=()
if [[ -n "$PRED_ROOT" ]]; then
    PRED_ROOT_ARGS=(--pred-root="$PRED_ROOT")
fi
python -m src.full_map.generate_full_hexel_map \
    --config="$CONFIG_FILE" \
    --output-dir="$OUTPUT_DIR" \
    "${PRED_ROOT_ARGS[@]}" \
    "${MOSAIC_ARG_ARRAY[@]}"

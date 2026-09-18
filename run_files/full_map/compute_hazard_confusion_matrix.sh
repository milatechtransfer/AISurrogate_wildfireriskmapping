#!/bin/bash
##SBATCH --mail-type=all
##SBATCH --mail-user=name@mila.quebec
#SBATCH --job-name=full_map_hazard_confusion
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mem=16Gb
#SBATCH --cpus-per-task=4

set -euo pipefail

# CPU-only and lightweight: both hazard class rasters are read in row-block windows and only
# the (num_classes x num_classes) confusion matrix is accumulated, so this never holds a full
# national-sized array in memory (unlike generate_national_hazard_map.sh) -- 16Gb is generous
# headroom for the row-block windows plus GDAL/rasterio overhead.

PRED_TIF=${1:-experiments/full_map/hazard_national_predicted_map.tif}
GT_TIF=${2:-experiments/full_map/hazard_national_ground_truth_map.tif}
OUTPUT_DIR=${OUTPUT_DIR:-experiments/full_map}
# Example: CONFUSION_ARGS="--save-plot --title='National Hazard Confusion Matrix'"
CONFUSION_ARGS=${CONFUSION_ARGS:-}

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
source .venv/bin/activate
export PYTHONUNBUFFERED=1

echo "Computing hazard confusion matrix: pred=$PRED_TIF gt=$GT_TIF"
read -r -a CONFUSION_ARG_ARRAY <<< "$CONFUSION_ARGS"
python -m src.full_map.compute_hazard_confusion_matrix \
    --pred-tif="$PRED_TIF" \
    --gt-tif="$GT_TIF" \
    --output-dir="$OUTPUT_DIR" \
    "${CONFUSION_ARG_ARRAY[@]}"

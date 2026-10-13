#!/bin/bash
##SBATCH --mail-type=all
##SBATCH --mail-user=name@mila.quebec
#SBATCH --job-name=full_map_predictions
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --mem=32Gb
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

set -euo pipefail

# Config file is required (e.g. configs/multi_output_spatial_weather.yaml); there is no
# universal default since it must match the checkpoint/model being evaluated.
CONFIG_FILE=${1:?Usage: sbatch generate_predictions.sh <config_file.yaml>}
# Example: PRED_ARGS="--stitch_mode=max --report_firezone_metrics"
PRED_ARGS=${PRED_ARGS:-}

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
source .venv/bin/activate
# Flush stdout/stderr immediately so log lines aren't lost if the job is
# preempted before Python's internal buffers would otherwise flush.
export PYTHONUNBUFFERED=1

# generate_predictions.py always forces config.logger.enabled = False, so no
# COMET_API_KEY check is needed here (unlike run_files/eval.sh / eval_hexels.sh).

echo "Generating full-map predictions (train+val+test) with config: $CONFIG_FILE"
# Parsed via eval (not `read -a`, which splits on whitespace and drops the embedded
# quotes) so e.g. --title='Burn Probability' survives as a single argument below.
eval "PRED_ARG_ARRAY=($PRED_ARGS)"
python -m src.full_map.generate_predictions \
    --config="$CONFIG_FILE" \
    "${PRED_ARG_ARRAY[@]}"

#!/bin/bash
#SBATCH --job-name=hazard_eval
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=unkillable
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --mem=32Gb
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1

set -euo pipefail

# Usage: sbatch run_files/eval_hazard.sh configs/<your_hazard_config>.yaml
# Extra CLI args (e.g. --metrics_only --skip_plots) can be passed via EVAL_ARGS:
#   EVAL_ARGS="--metrics_only" sbatch run_files/eval_hazard.sh configs/hazard_eval_common_input_pipeline.yaml
# Override mask_scope/save_dir/root_dir to run actual vs buffer variants from one config without output collisions:
#   EVAL_ARGS="--mask_scope buffer_only --root_dir /path/to/buffer_root --save_dir experiments/hazard_eval/buffer_only" \
#     sbatch run_files/eval_hazard.sh configs/hazard_eval_common_input_pipeline.yaml
CONFIG_FILE=${1:-configs/hazard_eval_common_input_pipeline.yaml}
EVAL_ARGS=${EVAL_ARGS:-}

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
source .venv/bin/activate

echo "Running hazard evaluation with config: $CONFIG_FILE"
EVAL_ARG_ARRAY=()
while IFS= read -r -d '' arg; do
    EVAL_ARG_ARRAY+=("$arg")
done < <(python - "$EVAL_ARGS" <<'PY'
import shlex
import sys

for arg in shlex.split(sys.argv[1]):
    sys.stdout.write(arg + "\0")
PY
)
python -m src.evaluate_hazard --config="$CONFIG_FILE" "${EVAL_ARG_ARRAY[@]}"

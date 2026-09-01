#!/bin/bash
#SBATCH --job-name=hybrid_v24_eval
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --constraint=ampere
#SBATCH --gres=gpu:a100:1

set -euo pipefail

repo_root="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$repo_root"
mkdir -p logs
source .venv/bin/activate

config="${CONFIG:-configs/mechanistic_hybrid_v24_reference.yaml}"
checkpoint="${CHECKPOINT:-/network/projects/amlrt/nrcan_wildfires/checkpoints/burnp3plus/final_experiments/mechanistic_hybrid_v24_native_512_crop_256_firesize_q3/best.pth}"
output_dir="${OUTPUT_DIR:-experiments/mechanistic/mechanistic_hybrid_v24_reference_evaluation}"

checkpoint=$(realpath "$checkpoint")
mkdir -p "$output_dir"
output_dir=$(realpath "$output_dir")
checkpoint_link="$output_dir/best.pth"
if [[ -e "$checkpoint_link" || -L "$checkpoint_link" ]]; then
    if [[ ! -L "$checkpoint_link" || "$(realpath "$checkpoint_link")" != "$checkpoint" ]]; then
        echo "ERROR: refusing to replace existing $checkpoint_link" >&2
        exit 1
    fi
else
    ln -s "$checkpoint" "$checkpoint_link"
fi

run_config=$(mktemp "${SLURM_TMPDIR:-/tmp}/hybrid_v24_eval.XXXXXX.yaml")
trap 'rm -f "$run_config"' EXIT
python - "$config" "$run_config" "$output_dir" <<'PY'
import sys
from pathlib import Path

import yaml

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
output_dir = Path(sys.argv[3])
with source.open() as handle:
    config = yaml.safe_load(handle)
config["save_dir"] = str(output_dir)
config["evaluation"]["checkpoint_filename"] = "best.pth"
config["logger"]["enabled"] = False
with destination.open("w") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY

read -r -a extra_eval_args <<< "${EVAL_ARGS:-}"
python -m src.evaluate_hexels \
    --config="$run_config" \
    --no_save_predictions \
    "${extra_eval_args[@]}"

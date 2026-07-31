#!/bin/bash
##SBATCH --mail-type=all
##SBATCH --mail-user=<your_email>@<your_domain>
#SBATCH --job-name=eval_hexel_benchmark
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --time=01:59:00
#SBATCH --mem-per-cpu=40Gb
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
set -euo pipefail

# ---------- Args ----------
# Single config: test_indices.csv already covers all 5 target hexels
# (01, 12, 16, 39, 49), so no per-hexel config files are needed --
# hex16 is isolated afterward from the per-hexel keys ("hex16/mse", etc.)
# already present in the metrics dict evaluate_hexels.py returns.
# NOTE: --fast_eval (added in the postprocessing-optimization update)
# already implies --metrics_only --skip_hexel_plots --no_save_predictions,
# so EVAL_ARGS defaults to just that single flag now.
CONFIG_FILE=${1:-configs/runtime_benchmarks/multi_output_hex16.yaml}
N_REPS=${N_REPS:-10}
EVAL_ARGS=${EVAL_ARGS:-"--tif_only"}

# NOTE: hardcoded rather than "cd ${SLURM_SUBMIT_DIR:-$(pwd)}" -- jobs are
# submitted from run_files/, and SLURM_SUBMIT_DIR resolves to wherever
# `sbatch` was invoked from, not the repo root where .venv/ actually lives.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
source .venv/bin/activate

echo "Using config: ${CONFIG_FILE}"
echo "Running ${N_REPS} repetitions for timing/memory variance"

# ---------- Run benchmark: N_REPS fresh process invocations ----------
# NOTE: each rep is a separate `python -m ...` call, not a Python-level loop --
# this avoids CUDA allocator/context state carrying over between reps, so
# timing and peak-memory numbers are independent trials rather than one
# progressively-warmed-up run.
for REP in $(seq 1 "$N_REPS"); do
    LOG_FILE="logs/eval_${SLURM_JOB_ID}_rep${REP}.out"
    echo "=======Starting rep ${REP}/${N_REPS}========"

    read -r -a EVAL_ARG_ARRAY <<< "$EVAL_ARGS"
    python -m src.evaluate_hexels \
        --config="$CONFIG_FILE" \
        "${EVAL_ARG_ARRAY[@]}" \
        > "$LOG_FILE" 2>&1

    echo "Rep ${REP} finished -> ${LOG_FILE}"
done

# ---------- Collect results ----------
echo "=======Benchmark Complete========"
grep -h "Total Evaluation Time\|Dataloader Setup Time\|Trainer Init Time\|Checkpoint Load Time\|Prediction Time\|Eval Time Excl. Dataloader Setup\|Peak Host RSS\|Peak GPU Reserved\|Peak GPU Allocated\|MPS Driver Allocated\|MPS Current Allocated" \
    logs/eval_${SLURM_JOB_ID}_rep*.out || echo "No timing/memory lines found -- check individual logs."

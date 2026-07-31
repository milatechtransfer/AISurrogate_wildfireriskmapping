#!/bin/bash
##SBATCH --mail-type=all
##SBATCH --mail-user=name@mila.quebec
#SBATCH --job-name=eval_hexel_benchmark_cpu
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=long-cpu
#SBATCH --ntasks=1
#SBATCH --time=01:59:00
#SBATCH --mem-per-cpu=40Gb
#SBATCH --cpus-per-task=4
set -euo pipefail

# ---------- Args ----------
# CPU-only counterpart to eval_hexel_benchmark.sh -- same config, same
# combined test_indices.csv (hexels 01, 12, 16, 39, 49), no --gres line so
# set_device() in src/utils.py falls through to "cpu" (confirmed: cuda ->
# mps -> cpu fallback chain, no GPU/MPS visible on a CPU-only node).
# NOTE: cpus-per-task/mem-per-cpu intentionally match the GPU script exactly --
# isolates device (GPU vs CPU) as the only variable, rather than also
# changing core count/memory at the same time. This does NOT match the
# 32-core BurnP3+ CPU baseline from the earlier table -- different
# comparison being made here (surrogate model GPU-vs-CPU parity).
# NOTE: --fast_eval (added in the postprocessing-optimization update)
# already implies --metrics_only --skip_hexel_plots --no_save_predictions.
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
echo "Running ${N_REPS} repetitions for timing/memory variance (CPU)"

# ---------- Run benchmark: N_REPS fresh process invocations ----------
# NOTE: each rep is a separate `python -m ...` call, not a Python-level loop --
# this avoids allocator/context state carrying over between reps, so
# timing and peak-memory numbers are independent trials rather than one
# progressively-warmed-up run.
for REP in $(seq 1 "$N_REPS"); do
    LOG_FILE="logs/eval_cpu_${SLURM_JOB_ID}_rep${REP}.out"
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
    logs/eval_cpu_${SLURM_JOB_ID}_rep*.out || echo "No timing/memory lines found -- check individual logs."

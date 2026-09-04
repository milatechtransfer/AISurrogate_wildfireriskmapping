#!/bin/bash
# Submit every hex16 counterfactual evaluation against the shared 256 -> 256 q3
# fire-size checkpoint, each with its plotting job chained via afterok.
#
# Usage:
#   bash run_files/counterfactual/submit_all_counterfactuals.sh [--dry-run] [name ...]
#
# With no names, all experiments below are submitted. Names are the keys of the
# EXPERIMENTS list (e.g. `fuel`, `wind_direction`).

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
mkdir -p logs

dry_run=0
requested=()
for arg in "$@"; do
    case "${arg}" in
        --dry-run) dry_run=1 ;;
        -*) echo "Unknown flag: ${arg}" >&2; exit 2 ;;
        *) requested+=("${arg}") ;;
    esac
done

# name : eval script : plot script
EXPERIMENTS=(
    "fuel:counterfactual_fuel_iROS.sh:counterfactual_fuel_plots.sh"
    "fuel_polygons:counterfactual_fuel_polygons_iROS.sh:counterfactual_fuel_polygons_plots.sh"
    "mean_weather:counterfactual_mean_weather_iROS.sh:counterfactual_mean_weather_plots.sh"
    "windy_weather:counterfactual_windy_weather_zone_dependent_iROS.sh:counterfactual_windy_weather_zone_dependent_plots.sh"
    "wind_direction:counterfactual_wind_direction_zone_dependent_iROS.sh:counterfactual_wind_direction_zone_dependent_plots.sh"
    "fire_size:counterfactual_fire_size_spread_days_iROS.sh:counterfactual_fire_size_spread_days_plots.sh"
)

run_dir="run_files/counterfactual"
selected=()
for entry in "${EXPERIMENTS[@]}"; do
    name="${entry%%:*}"
    if [[ ${#requested[@]} -eq 0 ]]; then
        selected+=("${entry}")
        continue
    fi
    for want in "${requested[@]}"; do
        if [[ "${want}" == "${name}" ]]; then
            selected+=("${entry}")
        fi
    done
done

if [[ ${#selected[@]} -eq 0 ]]; then
    echo "No matching experiments for: ${requested[*]}" >&2
    exit 2
fi

# Fail before submitting anything if a script is missing.
for entry in "${selected[@]}"; do
    IFS=':' read -r name eval_script plot_script <<<"${entry}"
    for script in "${eval_script}" "${plot_script}"; do
        if [[ ! -f "${run_dir}/${script}" ]]; then
            echo "Missing script: ${run_dir}/${script}" >&2
            exit 1
        fi
    done
done

printf '%-16s %-12s %-12s\n' "EXPERIMENT" "EVAL_JOB" "PLOT_JOB"
for entry in "${selected[@]}"; do
    IFS=':' read -r name eval_script plot_script <<<"${entry}"

    if [[ ${dry_run} -eq 1 ]]; then
        echo "sbatch ${run_dir}/${eval_script}"
        echo "sbatch --dependency=afterok:<eval> ${run_dir}/${plot_script}"
        printf '%-16s %-12s %-12s\n' "${name}" "DRYRUN" "DRYRUN"
        continue
    fi

    eval_job=$(sbatch --parsable "${run_dir}/${eval_script}")
    plot_job=$(sbatch --parsable --dependency="afterok:${eval_job}" --kill-on-invalid-dep=yes "${run_dir}/${plot_script}")
    printf '%-16s %-12s %-12s\n' "${name}" "${eval_job}" "${plot_job}"
done

"""
Aggregate multi-run (multi-seed) baseline evaluation results into a single CSV.

Reads back the metrics_{target}.json each seeded run writes to its own derived
save_dir (via apply_run_id_overrides), flattens the nested val/test/region-level
dicts into columns, concatenates across seeds, and appends mean/std summary rows.

Usage:
    python -m src.aggregate_baseline_multirun_results --config configs/baselines/bp_spatial_only_xgb.yaml --target=bp \
        --output_csv experiments/bp_spatial_only_xgb/multi_run_summary.csv
"""

import argparse
import json
import os

import pandas as pd

from src.config import SEEDS, apply_run_id_overrides
from src.train_tabular_baseline import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multi-run (multi-seed) baseline results into a CSV.")
    parser.add_argument(
        "--config", type=str, default="configs/baselines/bp_spatial_only_xgb.yaml", help="Path to the base YAML config used for training."
    )
    parser.add_argument(
        "--target",
        type=str,
        default="bp",
        choices=["bp", "fi", "ros"],
        help="Target the runs were trained on (must match the metrics_{target}.json filename).",
    )
    parser.add_argument(
        "--run_ids", type=int, nargs="+", default=list(range(len(SEEDS))), help="run_ids to aggregate. Defaults to all seeds."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to write the aggregated CSV. Defaults to '<base save_dir>/multi_run_summary_<target>.csv'.",
    )
    return parser.parse_args()


def flatten_metrics(target: str, run_id: int, seed: int, save_dir: str, metrics: dict) -> dict:
    """Flatten the nested metrics.json structure into a single-row dict for CSV export."""
    row = {"run_id": run_id, "seed": seed, "save_dir": save_dir}
    for split_key in ("val", "test"):
        for k, v in metrics.get(split_key, {}).items():
            row[f"{split_key}/patch/{k}"] = v
    for split_key in ("val_region_aggregated", "test_region_aggregated"):
        split_name = "val" if split_key.startswith("val") else "test"
        for k, v in metrics.get(split_key, {}).items():
            row[f"{split_name}/region/{k}"] = v
    return row


def collect_run_results(config_path: str, target: str, run_ids: list[int]) -> list[dict]:
    """Reads existing metrics_{target}.json for each run_id's derived save_dir.
    Does NOT re-run training -- assumes the seeded runs already completed via the
    SLURM array job (run_files/baselines/train_baseline_multi_run.sh)."""
    rows = []
    for run_id in run_ids:
        config = load_config(config_path)
        run_seed = apply_run_id_overrides(config, run_id)

        metrics_path = os.path.join(config.save_dir, f"metrics_{target}.json")
        if not os.path.isfile(metrics_path):
            raise FileNotFoundError(
                f"Missing {metrics_path} for run_id={run_id} (seed={run_seed}). " f"Has this seed's training run completed?"
            )
        with open(metrics_path) as f:
            metrics = json.load(f)

        rows.append(flatten_metrics(target, run_id, run_seed, config.save_dir, metrics))

    return rows


def build_summary_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)

    id_cols = ["run_id", "seed", "save_dir"]
    numeric_cols = df.select_dtypes(include="number").columns.difference(id_cols)
    summary = df[numeric_cols].agg(["mean", "std"])
    summary.insert(0, "save_dir", "")
    summary.insert(0, "seed", ["mean", "std"])
    summary.insert(0, "run_id", ["mean", "std"])

    return pd.concat([df, summary], ignore_index=True)


def main() -> None:
    args = parse_args()

    base_config = load_config(args.config)
    output_csv = args.output_csv or os.path.join(base_config.save_dir, f"multi_run_summary_{args.target}.csv")

    rows = collect_run_results(args.config, args.target, args.run_ids)
    df = build_summary_df(rows)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nWrote aggregated results for {len(df) - 2} run(s) to: {output_csv}")


if __name__ == "__main__":
    main()

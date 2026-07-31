"""
Aggregate multi-run (multi-seed) evaluation results into a single CSV.

Reads back each run_id's `test_metrics.csv` (written earlier by `src.train`'s test-set evaluation
on the best checkpoint, or by a standalone `python -m src.evaluate_hexels` call) from its derived
`save_dir` and concatenates them into one aggregated CSV with trailing mean/std summary rows. This
script never runs evaluation itself: run_ids whose folder or `test_metrics.csv` is missing are
skipped and simply excluded from aggregation.

Used to generate results tables for the paper submission (Tables 1, 7, 8, 11, 12)

Usage:
    python -m src.paper_utils.aggregate_multirun_results --config configs/multi_output_spatial_weather_spatial.yaml \
        --output_csv experiments/multi_output_spatial_weather_spatial/multi_run_eval_summary.csv
"""

import argparse
import os
import re

import pandas as pd

from src.config import SEEDS, apply_run_id_overrides
from src.evaluate_hexels import load_config

# Default per-target metrics/targets for the "by target output" summary table (one row per
# BP/FI/ROS target, restricted to the `test_hexel/all/` scope), matching the reporting-table
# style used for comparing model configurations (spatial / +weather / +weather+fire_size).
DEFAULT_TARGETS = ["bp", "fi", "ros"]
DEFAULT_TARGET_METRICS = ["normalized_mae", "normalized_bias", "spearman", "ccc", "auc_iou_top10", "auc_iou_full"]

METRIC_LABELS = {
    "normalized_mae": "Normalized MAE",
    "normalized_bias": "Normalized Bias",
    "spearman": "Spearman",
    "ccc": "CCC",
    "auc_iou_top10": "AUC IoU Top-10%",
    "auc_iou_full": "AUC IoU Full",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multi-run (multi-seed) evaluation results into a CSV.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/bp_common_input_pipeline.yaml",
        help="Path to the base YAML config file (the same one used for training with run_files/train_no_tmp_copy_array.sh).",
    )
    parser.add_argument(
        "--run_ids",
        type=int,
        nargs="+",
        default=list(range(len(SEEDS))),
        help=f"run_ids to evaluate and aggregate (each must be in [0, {len(SEEDS) - 1}]). Defaults to all seeds.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to write the aggregated CSV. Defaults to '<base config save_dir>/multi_run_eval_summary.csv'.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["test_patch_"],  # "test_patch_"
        help="Metric columns to keep and summarize with mean/std, e.g. 'test_patch_mse test_hexel/all/mse'. Matches exact "
        "column names or prefixes (e.g. 'test_hexel/all/' keeps every column starting with it). Defaults to every "
        "numeric column (all test_patch_*, test_hexel/*, and val_hexel/* metrics) when omitted.",
    )
    parser.add_argument(
        "--by_target_csv",
        type=str,
        default=None,
        help="Path to write the per-target-output (BP/FI/ROS) summary CSV, restricted to `--scope` and "
        "`--target_metrics`. Defaults to '<base config save_dir>/multi_run_eval_by_target.csv'. Pass "
        "'--by_target_csv=' (empty) to skip writing this file.",
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        default=DEFAULT_TARGETS,
        help=f"Target outputs to summarize one row each for in the by-target CSV. Defaults to {DEFAULT_TARGETS}.",
    )
    parser.add_argument(
        "--target_metrics",
        type=str,
        nargs="+",
        default=DEFAULT_TARGET_METRICS,
        help=f"Metrics to keep (per target) in the by-target CSV. Defaults to {DEFAULT_TARGET_METRICS}.",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["hexel", "patch", "per_hex"],
        default="hexel",
        help="Which metrics columns to read for the by-target CSV: 'hexel' (default) reads "
        "'test_hexel/<scope>/<target>_<metric>' columns; 'patch' reads 'test_patch_<target>/<metric>' columns "
        "instead ('--scope' is ignored in that case); 'per_hex' reads per-hexel "
        "'test_hexel/hex<id>/<target>_<metric>' columns and writes one block of rows per hex id (per-seed rows "
        "plus a trailing mean/std row for that hex), repeated for each target, all concatenated in one CSV "
        "('--scope' is ignored in that case). Only one source is used per run.",
    )
    parser.add_argument(
        "--hex_ids",
        type=str,
        nargs="+",
        default=None,
        help="Hex ids to include when '--source=per_hex' (e.g. '01 02 03'), matching the '<id>' in "
        "'test_hexel/hex<id>/...' columns. Defaults to every hex id found in the aggregated metrics.",
    )
    parser.add_argument(
        "--scope",
        type=str,
        default="all",
        help="test_hexel metric scope to read the by-target metrics from when '--source=hexel' (i.e. columns named "
        "'test_hexel/<scope>/<target>_<metric>'). Defaults to 'all'. Ignored when '--source=patch'.",
    )
    return parser.parse_args()


def find_available_save_dirs(config_path: str, run_ids: list[int]) -> list[str]:
    """
    Resolve each run_id's derived `save_dir` and keep only those whose `test_metrics.csv` already
    exists there (e.g. written earlier by `src.train`'s test-set evaluation on the best checkpoint,
    or by a standalone `src.evaluate_hexels` call). Never runs evaluation itself; run_ids missing a
    `test_metrics.csv` are skipped with a warning and simply excluded from aggregation.
    """
    save_dirs = []
    for run_id in run_ids:
        config = load_config(config_path)
        apply_run_id_overrides(config, run_id)

        test_metrics_csv = os.path.join(config.save_dir, "test_metrics.csv")
        if not os.path.isfile(test_metrics_csv):
            print(f"\n=== Skipping run_id={run_id}: {test_metrics_csv} not found (folder or test_metrics.csv missing) ===")
            continue

        save_dirs.append(config.save_dir)

    if not save_dirs:
        raise FileNotFoundError(f"No test_metrics.csv found for any of run_ids={run_ids} under config={config_path!r}.")

    return save_dirs


def read_multi_run_metrics(save_dirs: list[str]) -> pd.DataFrame:
    """
    Read each run's `test_metrics.csv` (written by `src.train` or `src.evaluate_hexels`) from
    `save_dirs` and concatenate them into a single per-seed DataFrame (one row per run, sorted by
    seed), with no column filtering or summary rows appended.
    """
    test_metrics_csvs = [os.path.join(save_dir, "test_metrics.csv") for save_dir in save_dirs]
    missing = [path for path in test_metrics_csvs if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Missing test_metrics.csv for run(s): {missing}")

    return pd.concat((pd.read_csv(path) for path in test_metrics_csvs), ignore_index=True).sort_values("seed").reset_index(drop=True)


def aggregate_eval_results(save_dirs: list[str], metrics: list[str] | None = None) -> pd.DataFrame:
    """
    Read each run's `test_metrics.csv` (written by `src.train` or `src.evaluate_hexels`) from
    `save_dirs`, concatenate them into a single DataFrame, and append trailing "mean"/"std"
    summary rows.

    If `metrics` is given, only id columns (run_id, seed, save_dir) plus columns matching
    `metrics` (by exact name or prefix, e.g. "test_hexel/all/") are kept and summarized. Otherwise
    every numeric column is kept and summarized.
    """
    df = read_multi_run_metrics(save_dirs)

    id_cols = ["run_id", "seed", "save_dir"]
    if metrics:
        keep_cols = id_cols + [col for col in df.columns if col not in id_cols and any(col == m or col.startswith(m) for m in metrics)]
        missing_metrics = [m for m in metrics if not any(col == m or col.startswith(m) for col in df.columns)]
        if missing_metrics:
            raise ValueError(f"No matching columns found for requested metrics: {missing_metrics}. Available columns: {list(df.columns)}")
        df = df[keep_cols]

    numeric_cols = df.select_dtypes(include="number").columns.difference(["run_id", "seed"])
    summary = df[numeric_cols].agg(["mean", "std"])
    summary.insert(0, "save_dir", "")
    summary.insert(0, "seed", ["mean", "std"])
    summary.insert(0, "run_id", ["mean", "std"])

    return pd.concat([df, summary], ignore_index=True)


def _by_target_column(source: str, target: str, metric: str, scope: str) -> str:
    """
    Build the source metrics CSV column name for a given (target, metric), matching how
    `src.evaluate_hexels` / `src.trainer` name columns for each metrics "source":
      - "hexel": `test_hexel/<scope>/<target>_<metric>` (default scope "all"), e.g.
        "test_hexel/all/bp_normalized_mae".
      - "patch": `test_patch_<target>/<metric>`, e.g. "test_patch_bp/normalized_mae".
    """
    if source == "hexel":
        return f"test_hexel/{scope}/{target}_{metric}"
    if source == "patch":
        return f"test_patch_{target}/{metric}"
    raise ValueError(f"Unsupported source={source!r}; expected 'hexel' or 'patch'.")


def discover_hex_ids(df: pd.DataFrame) -> list[str]:
    """
    Discover every distinct hex id present in a `test_metrics.csv`-derived DataFrame's per-hexel
    columns, i.e. every `<id>` in a `test_hexel/hex<id>/...` column name, sorted numerically when
    the ids are plain integers (e.g. "01", "02", ...), falling back to lexical order otherwise.
    """
    pattern = re.compile(r"^test_hexel/hex([^/]+)/")
    hex_ids = {match.group(1) for col in df.columns if (match := pattern.match(col))}

    def sort_key(hex_id: str) -> tuple[int, object]:
        try:
            return (0, int(hex_id))
        except ValueError:
            return (1, hex_id)

    return sorted(hex_ids, key=sort_key)


def summarize_by_hexel(
    df: pd.DataFrame,
    targets: list[str] = DEFAULT_TARGETS,
    target_metrics: list[str] = DEFAULT_TARGET_METRICS,
    hex_ids: list[str] | None = None,
) -> pd.DataFrame:
    """
    Reshape a per-seed multi-output metrics DataFrame (as returned by `read_multi_run_metrics`)
    into a long, per-hexel table: for each target output (BP/FI/ROS), and for each hex id (reading
    `test_hexel/hex<id>/<target>_<metric>` columns), one row per seed with the requested metric
    values, followed by a trailing "mean" row and a "std" row computed across that hex's seed rows.
    Blocks are concatenated hex-by-hex within each target, then target-by-target, into one CSV.

    `hex_ids` defaults to every hex id found in `df` (via `discover_hex_ids`).
    """
    if hex_ids is None:
        hex_ids = discover_hex_ids(df)
    if not hex_ids:
        raise ValueError("No per-hexel columns (e.g. 'test_hexel/hex01/...') found in aggregated metrics.")

    metric_labels = {metric: METRIC_LABELS.get(metric, metric) for metric in target_metrics}
    metric_cols = list(metric_labels.values())

    blocks = []
    for target in targets:
        for hex_id in hex_ids:
            scope = f"hex{hex_id}"
            cols = {metric: _by_target_column("hexel", target, metric, scope) for metric in target_metrics}
            missing = [col for col in cols.values() if col not in df.columns]
            if missing:
                raise ValueError(f"Missing expected column(s) {missing} in aggregated metrics. Available columns: {sorted(df.columns)}")

            block = pd.DataFrame({"target": target, "hex_id": hex_id, "seed": df["seed"]})
            for metric, col in cols.items():
                block[metric_labels[metric]] = pd.to_numeric(df[col], errors="coerce")

            mean_row: dict[str, object] = {"target": target, "hex_id": hex_id, "seed": "mean"}
            std_row: dict[str, object] = {"target": target, "hex_id": hex_id, "seed": "std"}
            for label in metric_cols:
                values = block[label].dropna()
                mean_row[label] = float(values.mean()) if len(values) else float("nan")
                std_row[label] = float(values.std(ddof=0)) if len(values) else float("nan")

            blocks.append(pd.concat([block, pd.DataFrame([mean_row, std_row])], ignore_index=True))

    return pd.concat(blocks, ignore_index=True)


def summarize_by_target(
    df: pd.DataFrame,
    targets: list[str] = DEFAULT_TARGETS,
    target_metrics: list[str] = DEFAULT_TARGET_METRICS,
    scope: str = "all",
    source: str = "hexel",
) -> pd.DataFrame:
    """
    Reshape a per-seed multi-output metrics DataFrame (as returned by `read_multi_run_metrics`)
    into a long table grouped by target output (BP/FI/ROS): for each target, one row per seed
    with the requested metric values, followed by a trailing "mean" row and a "std" row (computed
    across that target's seed rows).

    `source` selects which metrics columns to read: "hexel" (default) reads
    `test_hexel/<scope>/<target>_<metric>` columns (`scope` defaults to "all"); "patch" reads
    `test_patch_<target>/<metric>` columns instead (`scope` is ignored). Either way, only
    `target_metrics` are kept (default: normalized_mae, normalized_bias, spearman, ccc,
    auc_iou_top10, auc_iou_full).
    """
    metric_labels = {metric: METRIC_LABELS.get(metric, metric) for metric in target_metrics}

    blocks = []
    for target in targets:
        cols = {metric: _by_target_column(source, target, metric, scope) for metric in target_metrics}
        missing = [col for col in cols.values() if col not in df.columns]
        if missing:
            raise ValueError(f"Missing expected column(s) {missing} in aggregated metrics. Available columns: {sorted(df.columns)}")

        block = pd.DataFrame({"target": target, "seed": df["seed"]})
        for metric, col in cols.items():
            block[metric_labels[metric]] = pd.to_numeric(df[col], errors="coerce")

        metric_cols = list(metric_labels.values())
        mean_row: dict[str, object] = {"target": target, "seed": "mean"}
        std_row: dict[str, object] = {"target": target, "seed": "std"}
        for label in metric_cols:
            values = block[label].dropna()
            mean_row[label] = float(values.mean()) if len(values) else float("nan")
            std_row[label] = float(values.std(ddof=0)) if len(values) else float("nan")

        blocks.append(pd.concat([block, pd.DataFrame([mean_row, std_row])], ignore_index=True))

    return pd.concat(blocks, ignore_index=True)


def main() -> None:
    args = parse_args()

    base_config = load_config(args.config)
    output_csv = args.output_csv or os.path.join(base_config.save_dir, "multi_run_eval_summary.csv")
    by_target_suffix = "" if args.source == "hexel" else f"_{args.source}"
    by_target_csv = (
        args.by_target_csv
        if args.by_target_csv is not None
        else os.path.join(base_config.save_dir, f"multi_run_eval_by_target{by_target_suffix}.csv")
    )

    save_dirs = find_available_save_dirs(args.config, args.run_ids)

    df = aggregate_eval_results(save_dirs, metrics=args.metrics)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nWrote aggregated eval results for {len(df) - 2} run(s) to: {output_csv}")

    if by_target_csv:
        raw_df = read_multi_run_metrics(save_dirs)
        if args.source == "per_hex":
            by_target_df = summarize_by_hexel(
                raw_df,
                targets=args.targets,
                target_metrics=args.target_metrics,
                hex_ids=args.hex_ids,
            )
            summary_label = f"{by_target_df['hex_id'].nunique()} hex(es) across {len(args.targets)} target(s)"
        else:
            by_target_df = summarize_by_target(
                raw_df,
                targets=args.targets,
                target_metrics=args.target_metrics,
                scope=args.scope,
                source=args.source,
            )
            summary_label = f"{len(by_target_df)} target(s)"
        os.makedirs(os.path.dirname(by_target_csv) or ".", exist_ok=True)
        by_target_df.to_csv(by_target_csv, index=False)
        print(f"Wrote per-target-output summary ({summary_label}) to: {by_target_csv}")


if __name__ == "__main__":
    main()

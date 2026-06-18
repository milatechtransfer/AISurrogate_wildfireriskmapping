"""Paired summaries for fixed-model counterfactual predictions."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


@dataclass(frozen=True)
class PairedDeltaSummary:
    scenario: str
    endpoint: str
    hex_id: str
    n_pixels: int
    baseline_mean: float
    scenario_mean: float
    delta_mean: float
    delta_median: float
    delta_abs_mean: float
    delta_rmse: float
    delta_min: float
    delta_p05: float
    delta_p25: float
    delta_p75: float
    delta_p95: float
    delta_max: float
    delta_max_abs: float
    frac_delta_positive: float
    frac_delta_negative: float


def _valid_values(data: np.ma.MaskedArray | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.ma.asarray(data)
    values = np.asarray(arr.filled(np.nan), dtype=np.float64)
    valid = np.isfinite(values)
    return values, valid


def paired_delta_summary(
    baseline: np.ndarray | np.ma.MaskedArray,
    scenario: np.ndarray | np.ma.MaskedArray,
    *,
    scenario_name: str,
    endpoint: str,
    hex_id: str,
) -> PairedDeltaSummary:
    """Summarize scenario-baseline deltas over paired finite support."""

    baseline_values, baseline_valid = _valid_values(baseline)
    scenario_values, scenario_valid = _valid_values(scenario)
    valid = baseline_valid & scenario_valid
    if not valid.any():
        raise ValueError(f"No paired finite pixels for scenario={scenario_name!r}, endpoint={endpoint!r}, hex={hex_id!r}.")

    baseline_flat = baseline_values[valid]
    scenario_flat = scenario_values[valid]
    delta = scenario_flat - baseline_flat
    return PairedDeltaSummary(
        scenario=scenario_name,
        endpoint=endpoint,
        hex_id=hex_id,
        n_pixels=int(delta.size),
        baseline_mean=float(np.mean(baseline_flat)),
        scenario_mean=float(np.mean(scenario_flat)),
        delta_mean=float(np.mean(delta)),
        delta_median=float(np.median(delta)),
        delta_abs_mean=float(np.mean(np.abs(delta))),
        delta_rmse=float(np.sqrt(np.mean(delta**2))),
        delta_min=float(np.min(delta)),
        delta_p05=float(np.percentile(delta, 5)),
        delta_p25=float(np.percentile(delta, 25)),
        delta_p75=float(np.percentile(delta, 75)),
        delta_p95=float(np.percentile(delta, 95)),
        delta_max=float(np.max(delta)),
        delta_max_abs=float(np.max(np.abs(delta))),
        frac_delta_positive=float(np.mean(delta > 0.0)),
        frac_delta_negative=float(np.mean(delta < 0.0)),
    )


def _prediction_raster_path(prediction_dir: Path, hex_id: str) -> Path:
    return prediction_dir / "predicted_hexels" / f"hexel_{int(hex_id):02d}_predicted.tif"


def _read_raster(path: Path) -> np.ma.MaskedArray:
    if not path.exists():
        raise FileNotFoundError(path)
    with rasterio.open(path) as src:
        return src.read(1, masked=True)


def _index_by_scenario_endpoint(index: pd.DataFrame) -> dict[tuple[str, str], Path]:
    required = {"scenario", "endpoint", "prediction_dir"}
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"scenario_prediction_index.csv is missing required columns: {missing}")
    return {(str(row.scenario), str(row.endpoint)): Path(str(row.prediction_dir)) for row in index.itertuples(index=False)}


def compute_endpoint_delta_metrics(experiment_dir: Path, *, hex_id: str = "16") -> pd.DataFrame:
    """Compute paired raster-delta summaries for every non-baseline endpoint prediction."""

    index = pd.read_csv(experiment_dir / "scenario_prediction_index.csv")
    prediction_dirs = _index_by_scenario_endpoint(index)
    rows: list[dict] = []
    scenarios = sorted({scenario for scenario, _ in prediction_dirs if scenario != "baseline"})
    endpoints = sorted({endpoint for _, endpoint in prediction_dirs})
    for scenario in scenarios:
        for endpoint in endpoints:
            baseline_dir = prediction_dirs.get(("baseline", endpoint))
            scenario_dir = prediction_dirs.get((scenario, endpoint))
            if baseline_dir is None or scenario_dir is None:
                continue
            baseline = _read_raster(_prediction_raster_path(baseline_dir, hex_id))
            scenario_raster = _read_raster(_prediction_raster_path(scenario_dir, hex_id))
            summary = paired_delta_summary(
                baseline,
                scenario_raster,
                scenario_name=scenario,
                endpoint=endpoint,
                hex_id=hex_id,
            )
            rows.append(asdict(summary))
    return pd.DataFrame(rows)


def compute_hazard_delta_metrics(experiment_dir: Path, *, hex_id: str = "16") -> pd.DataFrame:
    """Compute paired summaries for Hazard = BP x FI where both endpoints exist."""

    index = pd.read_csv(experiment_dir / "scenario_prediction_index.csv")
    prediction_dirs = _index_by_scenario_endpoint(index)
    rows: list[dict] = []
    scenarios = sorted({scenario for scenario, _ in prediction_dirs if scenario != "baseline"})

    baseline_bp_dir = prediction_dirs.get(("baseline", "bp"))
    baseline_fi_dir = prediction_dirs.get(("baseline", "fi"))
    if baseline_bp_dir is None or baseline_fi_dir is None:
        return pd.DataFrame()

    baseline_bp = _read_raster(_prediction_raster_path(baseline_bp_dir, hex_id))
    baseline_fi = _read_raster(_prediction_raster_path(baseline_fi_dir, hex_id))
    baseline_hazard = np.ma.asarray(baseline_bp) * np.ma.asarray(baseline_fi)

    for scenario in scenarios:
        scenario_bp_dir = prediction_dirs.get((scenario, "bp"))
        scenario_fi_dir = prediction_dirs.get((scenario, "fi"))
        if scenario_bp_dir is None or scenario_fi_dir is None:
            continue
        scenario_bp = _read_raster(_prediction_raster_path(scenario_bp_dir, hex_id))
        scenario_fi = _read_raster(_prediction_raster_path(scenario_fi_dir, hex_id))
        scenario_hazard = np.ma.asarray(scenario_bp) * np.ma.asarray(scenario_fi)
        summary = paired_delta_summary(
            baseline_hazard,
            scenario_hazard,
            scenario_name=scenario,
            endpoint="hazard_bp_x_fi",
            hex_id=hex_id,
        )
        rows.append(asdict(summary))
    return pd.DataFrame(rows)


def write_counterfactual_metrics(experiment_dir: Path, *, hex_id: str = "16") -> tuple[Path, Path]:
    endpoint_metrics = compute_endpoint_delta_metrics(experiment_dir, hex_id=hex_id)
    hazard_metrics = compute_hazard_delta_metrics(experiment_dir, hex_id=hex_id)

    endpoint_path = experiment_dir / "counterfactual_endpoint_metrics.csv"
    hazard_path = experiment_dir / "counterfactual_hazard_metrics.csv"
    endpoint_metrics.to_csv(endpoint_path, index=False)
    hazard_metrics.to_csv(hazard_path, index=False)
    return endpoint_path, hazard_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute paired counterfactual prediction summaries.")
    parser.add_argument("--experiment_dir", type=Path, default=Path("experiments/counterfactual_hex16"))
    parser.add_argument("--hex_id", default="16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    endpoint_path, hazard_path = write_counterfactual_metrics(args.experiment_dir, hex_id=str(args.hex_id).zfill(2))
    print(f"Wrote endpoint metrics: {endpoint_path}")
    print(f"Wrote hazard metrics: {hazard_path}")


if __name__ == "__main__":
    main()

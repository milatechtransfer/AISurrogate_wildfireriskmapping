"""Build train-normalized per-hex ignition-count conditioning features."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from data_preparation.paths import MASK_SCOPE_CHOICES, Paths
from data_preparation.process_spread_opportunity import (
    Pmf,
    firezone_area_fractions,
    normalize_pmf,
    read_firezone_mapping,
    read_scenario_distributions,
    read_train_hex_ids,
)
from data_preparation.utils import find_hex_ids

logger = logging.getLogger(__name__)

RAW_MEAN_COLUMN = "IGNITION_COUNT_MEAN"
RAW_SD_COLUMN = "IGNITION_COUNT_SD"
RAW_CV_COLUMN = "IGNITION_COUNT_CV"
RAW_LOG_MEAN_COLUMN = "LOG1P_IGNITION_COUNT_MEAN"
NORMALIZED_LOG_MEAN_COLUMN = "NORM_LOG1P_IGNITION_COUNT_MEAN"
NORMALIZED_CV_COLUMN = "NORM_IGNITION_COUNT_CV"


def read_ignition_count_pmf(path: Path, scenario_frame: pd.DataFrame) -> Pmf:
    """Resolve fixed, uniformly sampled, or named user-defined ignition counts."""
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]

    if "DistributionType" in frame.columns:
        names = [str(value).strip() for value in frame["DistributionType"].dropna() if str(value).strip()]
    else:
        names = []

    unique_names = sorted(set(names))
    if unique_names:
        if len(unique_names) != 1:
            raise ValueError(f"Ignition-count table {path} references multiple distributions: {unique_names}")
        name = unique_names[0]
        if name.lower() in {"normal", "gamma"}:
            raise ValueError(f"Built-in ignition-count distribution {name!r} is not supported by this processor.")
        rows = scenario_frame[scenario_frame["Name"].eq(name)]
        if rows.empty:
            raise ValueError(f"Ignition-count distribution {name!r} referenced by {path} was not found.")
        pmf = normalize_pmf(
            zip(rows["Value"].astype(float), rows["RelativeFrequency"].astype(float), strict=True),
            context=f"{path} distribution {name!r}",
        )
    else:
        if "Mean" not in frame.columns:
            raise ValueError(f"Ignition-count table {path} contains neither a distribution nor a 'Mean' column.")
        values = pd.to_numeric(frame["Mean"], errors="coerce").dropna().to_numpy(dtype=np.float64)
        pmf = normalize_pmf(((float(value), 1.0) for value in values), context=f"{path} uniformly sampled means")

    if not pmf:
        raise ValueError(f"Ignition-count table {path} produced an empty distribution.")
    if any(value <= 0.0 or not float(value).is_integer() for value in pmf):
        raise ValueError(f"Ignition-count distribution from {path} must contain positive integer support: {sorted(pmf)}")
    return pmf


def ignition_count_statistics(pmf: Mapping[float, float]) -> dict[str, float]:
    """Return scale and relative-dispersion statistics for an ignition-count PMF."""
    mean = sum(float(value) * float(probability) for value, probability in pmf.items())
    variance = sum(float(probability) * (float(value) - mean) ** 2 for value, probability in pmf.items())
    sd = float(np.sqrt(max(variance, 0.0)))
    if mean <= 0.0:
        raise ValueError(f"Ignition-count mean must be positive, got {mean}.")
    return {
        RAW_MEAN_COLUMN: mean,
        RAW_SD_COLUMN: sd,
        RAW_CV_COLUMN: sd / mean,
        RAW_LOG_MEAN_COLUMN: float(np.log1p(mean)),
    }


def build_hex_ignition_count_rows(raw_root: Path, hex_id: str, *, mask_scope: str = "actual") -> pd.DataFrame:
    """Repeat one hex-level count summary over every fire-zone ID in that hex."""
    paths = Paths(hex_id=hex_id, root_dir=raw_root)
    scenario_frame = read_scenario_distributions(paths.scenario_distributions_table(hex_id))
    pmf = read_ignition_count_pmf(paths.ignition_count_table(hex_id), scenario_frame)
    statistics = ignition_count_statistics(pmf)
    _, id_to_name = read_firezone_mapping(paths.firezones_table(hex_id))
    area_fractions = firezone_area_fractions(
        raw_root=raw_root,
        hex_id=hex_id,
        id_to_name=id_to_name,
        mask_scope=mask_scope,
    )
    zone_ids = sorted(set(id_to_name) | set(area_fractions))
    if not zone_ids:
        raise ValueError(f"Hex{hex_id} has no fire-zone IDs for ignition-count spatialization.")
    return pd.DataFrame(
        [
            {
                "hex_id": int(hex_id),
                "GRIDCODE": zone_id,
                **statistics,
            }
            for zone_id in zone_ids
        ]
    )


def _fit_minmax(values: pd.Series, *, label: str) -> tuple[float, float]:
    minimum = float(values.min())
    maximum = float(values.max())
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        raise ValueError(f"Invalid {label} normalization range: minimum={minimum}, maximum={maximum}")
    return minimum, maximum


def save_normalization_params(
    path: Path,
    *,
    log_mean_range: tuple[float, float],
    cv_range: tuple[float, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "log1p_mean_minimum": log_mean_range[0],
        "log1p_mean_maximum": log_mean_range[1],
        "cv_minimum": cv_range[0],
        "cv_maximum": cv_range[1],
        "fit_scope": "unique training hexes",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_normalization_params(path: Path) -> tuple[tuple[float, float], tuple[float, float]]:
    payload = json.loads(path.read_text())
    return (
        (float(payload["log1p_mean_minimum"]), float(payload["log1p_mean_maximum"])),
        (float(payload["cv_minimum"]), float(payload["cv_maximum"])),
    )


def add_normalized_count_columns(
    frame: pd.DataFrame,
    *,
    train_hex_ids: set[int],
    norm_params_path: Path | None,
) -> tuple[pd.DataFrame, tuple[tuple[float, float], tuple[float, float]]]:
    """Normalize count scale and dispersion using each training hex exactly once."""
    if norm_params_path is not None and norm_params_path.exists():
        log_mean_range, cv_range = load_normalization_params(norm_params_path)
    else:
        unique_hexes = frame.drop_duplicates("hex_id")
        fit_frame = unique_hexes[unique_hexes["hex_id"].isin(train_hex_ids)]
        if fit_frame.empty:
            raise ValueError(f"No ignition-count rows match training hex IDs: {sorted(train_hex_ids)}")
        log_mean_range = _fit_minmax(fit_frame[RAW_LOG_MEAN_COLUMN], label="log1p ignition-count mean")
        cv_range = _fit_minmax(fit_frame[RAW_CV_COLUMN], label="ignition-count CV")
        if norm_params_path is not None:
            save_normalization_params(norm_params_path, log_mean_range=log_mean_range, cv_range=cv_range)

    output = frame.copy()
    output[NORMALIZED_LOG_MEAN_COLUMN] = (output[RAW_LOG_MEAN_COLUMN] - log_mean_range[0]) / (log_mean_range[1] - log_mean_range[0])
    output[NORMALIZED_CV_COLUMN] = (output[RAW_CV_COLUMN] - cv_range[0]) / (cv_range[1] - cv_range[0])
    return output, (log_mean_range, cv_range)


def build_ignition_count_table(
    *,
    raw_root: Path,
    output_path: Path,
    train_split_path: Path,
    norm_params_path: Path,
    mask_scope: str = "actual",
) -> pd.DataFrame:
    hex_ids = sorted(find_hex_ids(str(raw_root)))
    if not hex_ids:
        raise FileNotFoundError(f"No hex directories found under {raw_root}.")
    frame = pd.concat(
        [build_hex_ignition_count_rows(raw_root, hex_id, mask_scope=mask_scope) for hex_id in hex_ids],
        ignore_index=True,
    )
    if frame.duplicated(["hex_id", "GRIDCODE"]).any():
        duplicates = frame[frame.duplicated(["hex_id", "GRIDCODE"], keep=False)]
        raise ValueError(f"Duplicate ignition-count rows:\n{duplicates.to_string(index=False)}")

    frame, ranges = add_normalized_count_columns(
        frame,
        train_hex_ids=read_train_hex_ids(train_split_path),
        norm_params_path=norm_params_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    logger.info(
        "Saved %d ignition-count rows for %d hexes to %s; log-mean range=%s, CV range=%s.",
        len(frame),
        frame["hex_id"].nunique(),
        output_path,
        ranges[0],
        ranges[1],
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_root", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--train_split_path", type=Path, required=True)
    parser.add_argument("--norm_params_path", type=Path, required=True)
    parser.add_argument("--mask_scope", choices=MASK_SCOPE_CHOICES, default="actual")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    build_ignition_count_table(
        raw_root=args.raw_root,
        output_path=args.output_path,
        train_split_path=args.train_split_path,
        norm_params_path=args.norm_params_path,
        mask_scope=args.mask_scope,
    )


if __name__ == "__main__":
    main()

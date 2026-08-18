"""Build per-hex, per-FRU spread-opportunity distributions from NRCan scenarios."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from data_preparation.paths import MASK_SCOPE_CHOICES, Paths, normalize_mask_scope
from data_preparation.spatial.utils import load_spatial_raster
from data_preparation.utils import find_hex_ids

logger = logging.getLogger(__name__)

QUANTILES = (0.1, 0.5, 0.9)
RAW_MEAN_COLUMN = "TOTAL_BURN_HOURS_MEAN"
RAW_QUANTILE_COLUMNS = {
    0.1: "TOTAL_BURN_HOURS_Q10",
    0.5: "TOTAL_BURN_HOURS_Q50",
    0.9: "TOTAL_BURN_HOURS_Q90",
}
NORMALIZED_MEAN_COLUMN = "NORM_TOTAL_BURN_HOURS_MEAN"
NORMALIZED_QUANTILE_COLUMNS = {
    0.1: "NORM_TOTAL_BURN_HOURS_Q10",
    0.5: "NORM_TOTAL_BURN_HOURS_Q50",
    0.9: "NORM_TOTAL_BURN_HOURS_Q90",
}

Pmf = dict[float, float]


def normalize_pmf(values: Mapping[float, float] | Iterable[tuple[float, float]], *, context: str) -> Pmf:
    """Normalize non-negative relative weights into a discrete PMF."""
    items = values.items() if isinstance(values, Mapping) else values
    combined: dict[float, float] = defaultdict(float)
    for raw_value, raw_weight in items:
        value = float(raw_value)
        weight = float(raw_weight)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{context} contains invalid support value {raw_value!r}.")
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(f"{context} contains invalid probability weight {raw_weight!r}.")
        if weight > 0.0:
            combined[value] += weight

    total = sum(combined.values())
    if total <= 0.0:
        return {}
    return {value: combined[value] / total for value in sorted(combined)}


def multiply_pmfs(left: Mapping[float, float], right: Mapping[float, float], *, context: str) -> Pmf:
    """Return the PMF of the product of two independent discrete variables."""
    products: dict[float, float] = defaultdict(float)
    for left_value, left_probability in left.items():
        for right_value, right_probability in right.items():
            products[float(left_value) * float(right_value)] += float(left_probability) * float(right_probability)
    return normalize_pmf(products, context=context)


def mix_pmfs(weighted_pmfs: Iterable[tuple[float, Mapping[float, float]]], *, context: str) -> Pmf:
    """Mix complete PMFs, normalizing the supplied component weights."""
    supplied_components = [(float(weight), pmf) for weight, pmf in weighted_pmfs]
    if any(not np.isfinite(weight) or weight < 0.0 for weight, _ in supplied_components):
        raise ValueError(f"{context} contains an invalid mixture weight.")
    components = [(weight, pmf) for weight, pmf in supplied_components if weight > 0.0 and pmf]
    if not components:
        raise ValueError(f"{context} has no positive-weight PMF components.")

    total_weight = sum(weight for weight, _ in components)
    mixture: dict[float, float] = defaultdict(float)
    for weight, pmf in components:
        normalized_weight = weight / total_weight
        for value, probability in pmf.items():
            mixture[float(value)] += normalized_weight * float(probability)
    return normalize_pmf(mixture, context=context)


def discrete_quantile(pmf: Mapping[float, float], quantile: float) -> float:
    """Return the smallest support value whose CDF reaches ``quantile``."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"Quantile must lie in [0, 1], got {quantile}.")
    cumulative = 0.0
    for value, probability in sorted(pmf.items()):
        cumulative += float(probability)
        if cumulative + 1e-12 >= quantile:
            return float(value)
    raise ValueError(f"PMF has total probability {cumulative}, which does not reach quantile {quantile}.")


def pmf_mean(pmf: Mapping[float, float]) -> float:
    return sum(float(value) * float(probability) for value, probability in pmf.items())


def read_scenario_distributions(path: Path) -> pd.DataFrame:
    """Read the named discrete distributions, including the known hex28 header typo."""
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    if "Name" not in frame.columns:
        if len(frame.columns) == 3 and list(frame.columns[1:]) == ["Value", "RelativeFrequency"]:
            malformed_name_column = frame.columns[0]
            logger.warning("Using malformed first column %r as 'Name' in %s.", malformed_name_column, path)
            frame = frame.rename(columns={malformed_name_column: "Name"})
        else:
            raise ValueError(f"Unexpected scenario-distribution columns in {path}: {list(frame.columns)}")

    required_columns = ["Name", "Value", "RelativeFrequency"]
    missing_columns = set(required_columns) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Scenario-distribution table {path} is missing columns: {sorted(missing_columns)}")
    frame = frame[required_columns].copy()
    frame["Name"] = frame["Name"].astype(str).str.strip()
    frame["Value"] = pd.to_numeric(frame["Value"], errors="raise")
    frame["RelativeFrequency"] = pd.to_numeric(frame["RelativeFrequency"], errors="raise")
    if (frame["RelativeFrequency"] < 0.0).any():
        raise ValueError(f"Scenario-distribution table {path} contains negative relative frequencies.")
    return frame


def _named_distribution(frame: pd.DataFrame, name: str, *, context: str) -> Pmf | None:
    rows = frame[frame["Name"].eq(name)]
    if rows.empty:
        return None
    return normalize_pmf(
        zip(rows["Value"].astype(float), rows["RelativeFrequency"].astype(float), strict=True),
        context=context,
    )


def _row_distribution(row: pd.Series, scenario_frame: pd.DataFrame, *, context: str) -> Pmf | None:
    mean = row.get("Mean")
    if pd.notna(mean) and str(mean).strip():
        value = float(mean)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{context} contains non-positive mean {mean!r}.")
        return {value: 1.0}

    distribution_name = row.get("DistributionType")
    if pd.isna(distribution_name) or not str(distribution_name).strip():
        return None
    name = str(distribution_name).strip()
    pmf = _named_distribution(scenario_frame, name, context=f"{context} distribution {name!r}")
    if pmf is not None and any(value <= 0.0 for value in pmf):
        raise ValueError(f"{context} distribution {name!r} contains non-positive support.")
    return pmf


def read_daily_burning_hour_pmfs(path: Path, scenario_frame: pd.DataFrame) -> dict[str, Pmf | None]:
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    if "Season" not in frame.columns:
        raise ValueError(f"Daily-burning-hours table {path} is missing the 'Season' column.")

    result: dict[str, Pmf | None] = {}
    for _, row in frame.iterrows():
        season = str(row.get("Season", "")).strip()
        if not season:
            logger.warning("Ignoring blank season in daily-burning-hours table %s.", path)
            continue
        if season in result:
            raise ValueError(f"Daily-burning-hours table {path} contains duplicate season {season!r}.")
        result[season] = _row_distribution(row, scenario_frame, context=f"{path} season={season}")
    return result


def read_spread_day_pmfs(path: Path, scenario_frame: pd.DataFrame) -> dict[tuple[str, str], Pmf | None]:
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    required_columns = {"Season", "FireZone"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Spread-event-days table {path} is missing columns: {sorted(missing_columns)}")

    result: dict[tuple[str, str], Pmf | None] = {}
    for _, row in frame.iterrows():
        season = str(row.get("Season", "")).strip()
        firezone = str(row.get("FireZone", "")).strip()
        if not season or not firezone:
            logger.warning("Ignoring blank season/firezone row in spread-event-days table %s.", path)
            continue
        key = (firezone, season)
        if key in result:
            raise ValueError(f"Spread-event-days table {path} contains duplicate key {key}.")
        result[key] = _row_distribution(row, scenario_frame, context=f"{path} firezone={firezone}, season={season}")
    return result


def read_zone_season_likelihoods(path: Path) -> dict[tuple[str, str], float]:
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    required_columns = {"Season", "FireZone", "RelativeLikelihood"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Ignition-distribution table {path} is missing columns: {sorted(missing_columns)}")
    frame["RelativeLikelihood"] = pd.to_numeric(frame["RelativeLikelihood"], errors="raise")
    if (frame["RelativeLikelihood"] < 0.0).any():
        raise ValueError(f"Ignition-distribution table {path} contains negative likelihoods.")

    likelihoods: dict[tuple[str, str], float] = defaultdict(float)
    for _, row in frame.iterrows():
        season = str(row.get("Season", "")).strip()
        firezone = str(row.get("FireZone", "")).strip()
        if not season or not firezone:
            logger.warning("Ignoring blank season/firezone row in ignition-distribution table %s.", path)
            continue
        likelihoods[(firezone, season)] += float(row["RelativeLikelihood"])
    return dict(likelihoods)


def read_firezone_mapping(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    if not {"Name", "ID"} <= set(frame.columns):
        raise ValueError(f"Fire-zone table {path} must contain 'Name' and 'ID': {list(frame.columns)}")

    name_to_id: dict[str, int] = {}
    id_to_name: dict[int, str] = {}
    for _, row in frame.iterrows():
        name = str(row["Name"]).strip()
        try:
            raw_id = float(row["ID"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Fire-zone table {path} contains invalid row: {row.to_dict()}") from exc
        if not name or not np.isfinite(raw_id):
            raise ValueError(f"Fire-zone table {path} contains invalid row: {row.to_dict()}")
        zone_id = int(round(raw_id))
        if not np.isclose(raw_id, zone_id):
            raise ValueError(f"Fire-zone table {path} contains invalid row: {row.to_dict()}")
        if name in name_to_id or zone_id in id_to_name:
            raise ValueError(f"Fire-zone table {path} contains duplicate name or ID for {name!r}/{zone_id}.")
        name_to_id[name] = zone_id
        id_to_name[zone_id] = name
    return name_to_id, id_to_name


def firezone_area_fractions(
    *,
    raw_root: Path,
    hex_id: str,
    id_to_name: Mapping[int, str],
    mask_scope: str,
) -> dict[int, float]:
    paths = Paths(hex_id=hex_id, root_dir=raw_root)
    scope = normalize_mask_scope(mask_scope)
    raster, _ = load_spatial_raster(
        paths.firezones_grid(hex_id),
        reproject_flag=False,
        mask_path=paths.mask_grid(hex_id, mask_scope=scope),
    )
    values = np.asarray(raster.compressed(), dtype=np.float64)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        raise ValueError(f"Hex{hex_id} has no positive fire-zone pixels in mask scope {scope!r}.")
    rounded = np.rint(values)
    if not np.allclose(values, rounded, atol=1e-3):
        bad_value = float(values[np.argmax(np.abs(values - rounded))])
        raise ValueError(f"Hex{hex_id} fire-zone raster contains non-integer ID {bad_value}.")

    zone_ids, counts = np.unique(rounded.astype(np.int64), return_counts=True)
    unknown_ids = sorted(int(zone_id) for zone_id in zone_ids if int(zone_id) not in id_to_name)
    if unknown_ids:
        logger.warning(
            "Hex%s fire-zone raster contains IDs missing from FireZones.csv; they will receive the explicit hex fallback: %s",
            hex_id,
            unknown_ids,
        )
    total = int(counts.sum())
    return {int(zone_id): int(count) / total for zone_id, count in zip(zone_ids, counts, strict=True)}


def build_hex_spread_opportunity_from_tables(
    *,
    hex_id: str,
    spread_day_pmfs: Mapping[tuple[str, str], Pmf | None],
    daily_hour_pmfs: Mapping[str, Pmf | None],
    likelihoods: Mapping[tuple[str, str], float],
    name_to_id: Mapping[str, int],
    id_to_name: Mapping[int, str],
    area_fractions: Mapping[int, float],
) -> pd.DataFrame:
    """Build direct FRU mixtures and one area-times-likelihood fallback mixture."""
    components_by_zone: dict[str, list[tuple[float, Pmf]]] = defaultdict(list)
    fallback_components: list[tuple[float, Pmf]] = []
    dropped_likelihood: dict[str, float] = defaultdict(float)

    for (firezone, season), likelihood in sorted(likelihoods.items()):
        likelihood = float(likelihood)
        if likelihood <= 0.0:
            continue
        day_pmf = spread_day_pmfs.get((firezone, season))
        hour_pmf = daily_hour_pmfs.get(season)
        if not day_pmf or not hour_pmf:
            dropped_likelihood[firezone] += likelihood
            logger.warning(
                "Hex%s dropping positive scenario likelihood %.6g for firezone=%s, season=%s "
                "because spread days or daily hours are unavailable.",
                hex_id,
                likelihood,
                firezone,
                season,
            )
            continue

        total_hours = multiply_pmfs(day_pmf, hour_pmf, context=f"hex{hex_id} {firezone} {season} total burning hours")
        components_by_zone[firezone].append((likelihood, total_hours))
        zone_id = name_to_id.get(firezone)
        area_weight = area_fractions.get(zone_id, 0.0) if zone_id is not None else 0.0
        if area_weight > 0.0:
            fallback_components.append((area_weight * likelihood, total_hours))

    fallback_pmf = mix_pmfs(fallback_components, context=f"hex{hex_id} area-weighted fallback")
    rows: list[dict[str, float | int | str]] = []
    for zone_id, firezone in sorted(id_to_name.items()):
        components = components_by_zone.get(firezone, [])
        used_fallback = not components
        pmf = fallback_pmf if used_fallback else mix_pmfs(components, context=f"hex{hex_id} firezone={firezone} season mixture")
        row: dict[str, float | int | str] = {
            "hex_id": int(hex_id),
            "GRIDCODE": int(zone_id),
            "FIREZONE": firezone,
            RAW_MEAN_COLUMN: pmf_mean(pmf),
            "SCENARIO_FALLBACK": int(used_fallback),
            "SCENARIO_SEASON_COUNT": len(components),
            "SCENARIO_DROPPED_LIKELIHOOD": float(dropped_likelihood.get(firezone, 0.0)),
        }
        for quantile in QUANTILES:
            row[RAW_QUANTILE_COLUMNS[quantile]] = discrete_quantile(pmf, quantile)
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"Hex{hex_id} produced no spread-opportunity rows.")
    if frame.duplicated(["hex_id", "GRIDCODE"]).any():
        raise ValueError(f"Hex{hex_id} produced duplicate (hex_id, GRIDCODE) rows.")
    return frame


def build_hex_spread_opportunity(raw_root: Path, hex_id: str, *, mask_scope: str = "actual") -> pd.DataFrame:
    paths = Paths(hex_id=hex_id, root_dir=raw_root)
    scenario_frame = read_scenario_distributions(paths.scenario_distributions_table(hex_id))
    spread_day_pmfs = read_spread_day_pmfs(paths.spread_event_days_table(hex_id), scenario_frame)
    daily_hour_pmfs = read_daily_burning_hour_pmfs(paths.daily_burning_hours_table(hex_id), scenario_frame)
    likelihoods = read_zone_season_likelihoods(paths.ignition_distribution_table(hex_id))
    name_to_id, id_to_name = read_firezone_mapping(paths.firezones_table(hex_id))
    area_fractions = firezone_area_fractions(
        raw_root=raw_root,
        hex_id=hex_id,
        id_to_name=id_to_name,
        mask_scope=mask_scope,
    )
    for zone_id in area_fractions:
        if zone_id not in id_to_name:
            fallback_name = f"unmapped_gridcode_{zone_id}"
            id_to_name[zone_id] = fallback_name
            name_to_id[fallback_name] = zone_id
    return build_hex_spread_opportunity_from_tables(
        hex_id=hex_id,
        spread_day_pmfs=spread_day_pmfs,
        daily_hour_pmfs=daily_hour_pmfs,
        likelihoods=likelihoods,
        name_to_id=name_to_id,
        id_to_name=id_to_name,
        area_fractions=area_fractions,
    )


def read_train_hex_ids(path: Path) -> set[int]:
    if not path.exists():
        raise FileNotFoundError(f"Training split not found: {path}")
    frame = pd.read_csv(path)
    if "hex_id" not in frame.columns:
        raise ValueError(f"Training split {path} is missing the 'hex_id' column.")
    hex_ids = {int(value) for value in pd.to_numeric(frame["hex_id"], errors="raise").dropna()}
    if not hex_ids:
        raise ValueError(f"Training split {path} contains no hex IDs.")
    return hex_ids


def save_normalization_params(path: Path, *, minimum: float, maximum: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "minimum_total_burn_hours": float(minimum),
        "maximum_total_burn_hours": float(maximum),
        "fit_columns": [RAW_QUANTILE_COLUMNS[quantile] for quantile in QUANTILES],
        "shared_across_quantiles": True,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_normalization_params(path: Path) -> tuple[float, float]:
    payload = json.loads(path.read_text())
    return float(payload["minimum_total_burn_hours"]), float(payload["maximum_total_burn_hours"])


def add_shared_minmax_columns(
    frame: pd.DataFrame,
    *,
    train_hex_ids: set[int] | None,
    norm_params_path: Path | None,
) -> tuple[pd.DataFrame, tuple[float, float]]:
    """Apply one leakage-safe linear min-max transform to mean and all q3 values."""
    raw_columns = [RAW_QUANTILE_COLUMNS[quantile] for quantile in QUANTILES]
    if norm_params_path is not None and norm_params_path.exists():
        minimum, maximum = load_normalization_params(norm_params_path)
    else:
        fit_frame = frame if train_hex_ids is None else frame[frame["hex_id"].isin(train_hex_ids)]
        if fit_frame.empty:
            raise ValueError(f"No spread-opportunity rows match training hex IDs: {sorted(train_hex_ids or [])}")
        values = fit_frame[raw_columns].to_numpy(dtype=np.float64)
        minimum = float(np.nanmin(values))
        maximum = float(np.nanmax(values))
        if norm_params_path is not None:
            save_normalization_params(norm_params_path, minimum=minimum, maximum=maximum)
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        raise ValueError(f"Invalid spread-opportunity normalization range: minimum={minimum}, maximum={maximum}")

    output = frame.copy()
    scale = maximum - minimum
    output[NORMALIZED_MEAN_COLUMN] = (output[RAW_MEAN_COLUMN] - minimum) / scale
    for quantile in QUANTILES:
        output[NORMALIZED_QUANTILE_COLUMNS[quantile]] = (output[RAW_QUANTILE_COLUMNS[quantile]] - minimum) / scale
    return output, (minimum, maximum)


def build_spread_opportunity_table(
    *,
    raw_root: Path,
    output_path: Path,
    train_split_path: Path | None,
    norm_params_path: Path | None,
    mask_scope: str = "actual",
) -> pd.DataFrame:
    hex_ids = sorted(find_hex_ids(str(raw_root)))
    if not hex_ids:
        raise FileNotFoundError(f"No hex directories found under {raw_root}.")
    frames = [build_hex_spread_opportunity(raw_root, hex_id, mask_scope=mask_scope) for hex_id in hex_ids]
    frame = pd.concat(frames, ignore_index=True)
    if frame.duplicated(["hex_id", "GRIDCODE"]).any():
        duplicates = frame[frame.duplicated(["hex_id", "GRIDCODE"], keep=False)]
        raise ValueError(f"Duplicate spread-opportunity rows:\n{duplicates.to_string(index=False)}")

    train_hex_ids = read_train_hex_ids(train_split_path) if train_split_path is not None else None
    frame, normalization_range = add_shared_minmax_columns(
        frame,
        train_hex_ids=train_hex_ids,
        norm_params_path=norm_params_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    logger.info(
        "Saved %d spread-opportunity rows to %s using shared range [%.6g, %.6g]; fallbacks=%d.",
        len(frame),
        output_path,
        normalization_range[0],
        normalization_range[1],
        int(frame["SCENARIO_FALLBACK"].sum()),
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
    build_spread_opportunity_table(
        raw_root=args.raw_root,
        output_path=args.output_path,
        train_split_path=args.train_split_path,
        norm_params_path=args.norm_params_path,
        mask_scope=args.mask_scope,
    )


if __name__ == "__main__":
    main()

from pathlib import Path

import numpy as np
import pandas as pd

from data_preparation.paths import Paths
from data_preparation.utils import find_hex_ids

# Fuel encoding modes that use a per-pixel curve vector (as opposed to scalar
# or one-hot encodings).  Add new curve-based feature names here.
FUEL_CURVE_ENCODINGS: frozenset[str] = frozenset({"iROS", "HFI"})

# Single CSV produced by compute_vector_values_national.R containing all fuel curve columns.
# Configurable via GridParams/DataConfig.fuel_curves_filename; this is only the fallback default.
DEFAULT_FUEL_CURVES_CSV = "fbp_curves_national_fuel.csv"

# Maps feature_name -> value_column_name (all features are read from the same CSV file).
_FEATURE_COLUMN: dict[str, str] = {
    "iROS": "ROS",
    "HFI": "HFI",
}


def normalize_hex_id(hex_id: str | int) -> str:
    """
    Normalize hex IDs to two digits.

    Examples:
        1    -> "01"
        "1"  -> "01"
        "01" -> "01"
    """
    return str(hex_id).replace("hex", "").zfill(2)


# Keep the private alias for internal callers.
_normalize_hex_id = normalize_hex_id


def read_curves(
    ros_csv_path: str | Path,
    code_col: str = "fbp_code",
    season_col: str = "SeasonState",
    isi_col: str = "ISI",
    feature_col: str = "ROS",
) -> dict[int, dict[str, pd.Series]]:
    """
    Read the ROS curve CSV once.

    Returns
    -------
    dict
        {
            fbp_code: {
                SeasonState: pd.Series(
                    index=ISI,
                    values=ROS,
                )
            }
        }
    """
    ros_csv_path = Path(ros_csv_path)

    if not ros_csv_path.exists():
        raise FileNotFoundError(f"Fuel curve CSV not found: {ros_csv_path}")

    df = pd.read_csv(ros_csv_path)

    required_cols = {
        code_col,
        season_col,
        isi_col,
        feature_col,
    }

    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(f"Missing required columns in ROS CSV: {sorted(missing_cols)}")

    df = df[
        [
            code_col,
            season_col,
            isi_col,
            feature_col,
        ]
    ].copy()

    df[code_col] = pd.to_numeric(
        df[code_col],
        errors="raise",
    ).astype(int)

    df[isi_col] = pd.to_numeric(
        df[isi_col],
        errors="raise",
    )

    df[feature_col] = pd.to_numeric(
        df[feature_col],
        errors="raise",
    )

    df[season_col] = df[season_col].astype(str).str.strip()

    duplicate_mask = df.duplicated(
        subset=[
            code_col,
            season_col,
            isi_col,
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = (
            df.loc[
                duplicate_mask,
                [
                    code_col,
                    season_col,
                    isi_col,
                ],
            ]
            .drop_duplicates()
            .sort_values(
                [
                    code_col,
                    season_col,
                    isi_col,
                ]
            )
        )

        raise ValueError(
            f"Multiple feature values were found for the same fbp_code, SeasonState, and ISI:\n{duplicates.to_string(index=False)}"
        )

    curves_by_code: dict[int, dict[str, pd.Series]] = {}

    for fbp_code, code_group in df.groupby(
        code_col,
        sort=False,
    ):
        season_curves: dict[str, pd.Series] = {}

        for season_state, season_group in code_group.groupby(
            season_col,
            sort=False,
        ):
            curve = season_group.sort_values(isi_col).set_index(isi_col)[feature_col].astype(float)
            nan_isi = curve.index[curve.isna()].tolist()
            if nan_isi:
                raise ValueError(
                    f"Fuel curve CSV contains NaN {feature_col!r} values for "
                    f"fbp_code={int(fbp_code)}, SeasonState={str(season_state)!r} "
                    f"at ISI={nan_isi}. Regenerate the CSV from the R script."
                )

            season_curves[str(season_state)] = curve

        curves_by_code[int(fbp_code)] = season_curves

    return curves_by_code


def _read_hex_season_weights(
    distribution_path: str | Path,
    season_mapping: dict[str, str],
) -> dict[str, float]:
    """
    Read one hex ignition distribution file and calculate one
    normalized weight per ROS SeasonState.

    Example:

        ,,s1,Lightning,fru21,0.5
        ,,s2,Lightning,fru21,1.5

    Parameters
    ----------
    season_mapping
        Maps ignition seasons to ROS SeasonState values.

        Example:
            {
                "s1": "leafless",
                "s2": "green",
            }

    Returns
    -------
    dict
        Example:

            {
                "leafless": 0.09,
                "green": 0.91,
            }
    """
    distribution_path = Path(distribution_path)

    if not distribution_path.exists():
        raise FileNotFoundError(f"Ignition distribution CSV not found: {distribution_path}")

    distribution_df = pd.read_csv(distribution_path)

    if "Season" not in distribution_df.columns:
        raise ValueError(
            f"Ignition distribution CSV is missing a 'Season' column: {distribution_path}. "
            f"Found columns: {list(distribution_df.columns)}"
        )

    distribution_df["Season"] = distribution_df["Season"].astype(str).str.strip()

    distribution_df["RelativeLikelihood"] = pd.to_numeric(
        distribution_df["RelativeLikelihood"],
        errors="raise",
    )

    ignition_weights = distribution_df.groupby("Season")["RelativeLikelihood"].sum().to_dict()

    # Validate that all seasons in the ignition distribution are covered by the mapping.
    # A mismatch here means the GreenUp table uses different season names than the
    # ignition distribution (e.g. "s1"/"s2" vs "Spring"/"Summer-Fall").
    unmapped_seasons = set(ignition_weights) - set(season_mapping)
    if unmapped_seasons:
        raise ValueError(
            f"Ignition distribution {distribution_path} contains season(s) "
            f"{sorted(unmapped_seasons)} that have no entry in the GreenUp table. "
            f"GreenUp table maps: {sorted(season_mapping)}. "
            f"Ensure the GreenUp table uses the same season names as the ignition distribution."
        )

    season_state_weights: dict[str, float] = {}

    for ignition_season, season_state in season_mapping.items():
        weight = float(
            ignition_weights.get(
                ignition_season,
                0.0,
            )
        )

        season_state_weights[season_state] = (
            season_state_weights.get(
                season_state,
                0.0,
            )
            + weight
        )

    total_weight = sum(season_state_weights.values())

    if total_weight <= 0:
        raise ValueError(f"The ignition distribution contains no positive season weight: {distribution_path}")

    return {season_state: weight / total_weight for season_state, weight in season_state_weights.items()}


def _combine_season_curves(
    fbp_code: int,
    hex_id: str,
    season_curves: dict[str, pd.Series],
    season_weights: dict[str, float],
) -> np.ndarray:
    """
    Return one ROS vector for one fbp_code and one hex_id.

    If only one SeasonState exists, that curve is returned directly.

    If multiple SeasonState curves exist, they are aligned by ISI and
    combined using the hex-specific ignition season weights.
    """
    season_states = list(season_curves.keys())

    if not season_states:
        raise ValueError(f"No ROS curves found for fbp_code={fbp_code}")

    # Only one SeasonState exists for this code.
    if len(season_states) == 1:
        only_state = season_states[0]

        return season_curves[only_state].sort_index().to_numpy(dtype=np.float32)

    # Keep only the SeasonStates that this hex actually has ignition weight for.
    # A SeasonState absent from season_weights means it never occurs in this hex
    # (e.g. a fully green hex has no leafless weight), so it is simply excluded.
    season_states = [s for s in season_states if s in season_weights]

    if not season_states:
        raise ValueError(
            f"None of the ROS SeasonStates for fbp_code={fbp_code}, hex_id={hex_id} "
            f"have an ignition season weight. Available weights={season_weights}"
        )

    # If only one applicable state remains after filtering, return it directly.
    if len(season_states) == 1:
        return season_curves[season_states[0]].sort_index().to_numpy(dtype=np.float32)

    applicable_weights = {season_state: season_weights[season_state] for season_state in season_states}

    total_applicable_weight = sum(applicable_weights.values())

    if total_applicable_weight <= 0:
        raise ValueError(
            f"The applicable ignition season weights sum to zero for fbp_code={fbp_code}, hex_id={hex_id}. Weights={applicable_weights}"
        )

    # Re-normalize over only the states that apply to this FBP code.
    applicable_weights = {season_state: weight / total_applicable_weight for season_state, weight in applicable_weights.items()}

    # Align all curves using ISI as the index.
    aligned_curves = pd.concat(
        {season_state: season_curves[season_state] for season_state in season_states},
        axis=1,
    ).sort_index()

    if aligned_curves.isna().any().any():
        isi_per_state = {s: sorted(season_curves[s].index.tolist()) for s in season_states}
        missing_isi = aligned_curves.index[aligned_curves.isna().any(axis=1)].tolist()
        raise ValueError(
            f"SeasonState curves have mismatched ISI values for fbp_code={fbp_code}. "
            f"ISI values per state: {isi_per_state}. "
            f"ISI bins with at least one missing value: {missing_isi}. "
            f"Regenerate the fuel curve CSV from the R script."
        )

    weighted_ros = np.zeros(
        len(aligned_curves),
        dtype=np.float32,
    )

    for season_state, weight in applicable_weights.items():
        weighted_ros += np.float32(weight) * aligned_curves[season_state].to_numpy(dtype=np.float32)

    return weighted_ros


def _build_season_mapping_from_greenup(greenup_path: str | Path) -> dict[str, str]:
    """
    Build a season-name -> ROS SeasonState mapping from a GreenUp CSV.

    The CSV must have columns ``Season`` and ``GreenUp``.  A ``GreenUp``
    value of ``"Yes"`` (case-insensitive) maps to ``"green"``; anything
    else maps to ``"leafless"``.

    Example input
    -------------
    Season,GreenUp
    s1,No
    s2,Yes
    s3, Yes

    Example output
    --------------
    {"s1": "leafless", "s2": "green", "s3": "green"}
    """
    greenup_path = Path(greenup_path)

    if not greenup_path.exists():
        raise FileNotFoundError(f"GreenUp table not found: {greenup_path}")

    df = pd.read_csv(greenup_path)

    required_cols = {"Season", "GreenUp"}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(f"Missing required columns in GreenUp table {greenup_path}: {sorted(missing_cols)}")

    df["Season"] = df["Season"].astype(str).str.strip()
    df["GreenUp"] = df["GreenUp"].astype(str).str.strip()

    return {row["Season"]: "green" if row["GreenUp"].lower() == "yes" else "leafless" for _, row in df.iterrows()}


def build_fuel_curve_lookup(
    root_dir: str | Path,
    raw_data_dir: str | Path,
    code_col: str = "fbp_code",
    season_col: str = "SeasonState",
    isi_col: str = "ISI",
    feature_name: str = "iROS",
    fuel_curves_filename: str = DEFAULT_FUEL_CURVES_CSV,
) -> dict[tuple[int, str | None], np.ndarray]:
    """
    Construct the complete fuel curve lookup table once at runtime.

    Lookup format
    -------------
    Hex-specific (multi-season fuel codes whose blend depends on ignition weights):

        lookup[(fbp_code, hex_id)] -> curve vector

    Hex-independent (single-SeasonState codes such as ``direct``, ``nonfuel``,
    ``water``, or codes with only one seasonal state):

        lookup[(fbp_code, None)] -> curve vector

    Example
    -------
        curve_lookup[(13, "01")]   # seasonal, hex-specific
        curve_lookup[(1,  None)]   # direct, shared across all hexes

    For each hex, its ignition distribution file is obtained using:

        all_paths = Paths(
            hex_id=hex_id,
            root_dir=root_dir,
        )

        ign_csv = all_paths.ignition_distribution_table(
            hex_id=hex_id
        )

    Parameters
    ----------
    root_dir
        Path to data directory containing the fuel curve CSV.

    raw_data_dir
        Root directory of hexel data.

    feature_name
        Curve feature to load.  Must be a key in ``_FEATURE_COLUMN``
        (e.g. ``"iROS"`` or ``"HFI"``).

    fuel_curves_filename
        Filename (relative to root_dir) of the fuel curve CSV produced by
        compute_vector_values_national.R. Defaults to ``DEFAULT_FUEL_CURVES_CSV``.
    """
    if feature_name not in _FEATURE_COLUMN:
        raise ValueError(f"Unknown feature_name={feature_name!r}. Supported values: {sorted(_FEATURE_COLUMN)}")

    feature_col = _FEATURE_COLUMN[feature_name]

    # Read and prepare all curves only once.
    curves_by_code = read_curves(
        ros_csv_path=Path(root_dir) / fuel_curves_filename,
        code_col=code_col,
        season_col=season_col,
        isi_col=isi_col,
        feature_col=feature_col,
    )
    curve_lookup: dict[tuple[int, str | None], np.ndarray] = {}

    # Split codes into hex-independent (single SeasonState) and hex-specific
    # (multiple SeasonStates whose blend varies with ignition season weights).
    single_state_codes = {code: curves for code, curves in curves_by_code.items() if len(curves) == 1}
    multi_state_codes = {code: curves for code, curves in curves_by_code.items() if len(curves) > 1}

    # Single-state codes produce the same vector for every hex — store once.
    for fbp_code, season_curves in single_state_codes.items():
        only_state = next(iter(season_curves))
        curve_lookup[(fbp_code, None)] = season_curves[only_state].sort_index().to_numpy(dtype=np.float32)

    # Multi-state codes must be blended per hex using hex-specific season weights.
    hex_ids = find_hex_ids(str(raw_data_dir))

    for raw_hex_id in hex_ids:
        hex_id = _normalize_hex_id(raw_hex_id)

        all_paths = Paths(
            hex_id=hex_id,
            root_dir=raw_data_dir,
        )

        greenup_path = all_paths.seasons_greenup_table(hex_id=hex_id)
        hex_season_mapping = _build_season_mapping_from_greenup(greenup_path)

        ignition_distribution_path = Path(all_paths.ignition_distribution_table(hex_id=hex_id))

        season_weights = _read_hex_season_weights(
            distribution_path=ignition_distribution_path,
            season_mapping=hex_season_mapping,
        )

        for fbp_code, season_curves in multi_state_codes.items():
            curve_lookup[(fbp_code, hex_id)] = _combine_season_curves(
                fbp_code=fbp_code,
                hex_id=hex_id,
                season_curves=season_curves,
                season_weights=season_weights,
            )

    return curve_lookup


def get_fuel_curve_from_lookup(
    curve_lookup: dict[tuple[int, str | None], np.ndarray],
    fbp_code: int,
    hex_id: str | int,
    copy: bool = False,
) -> np.ndarray:
    """
    Retrieve one vector from the precomputed lookup.

    For hex-specific (multi-season) codes the key is ``(fbp_code, hex_id)``.
    For hex-independent (single-state) codes the key is ``(fbp_code, None)``;
    the lookup falls back to that sentinel automatically.
    """
    normalized_hex_id = _normalize_hex_id(hex_id)

    hex_key = (int(fbp_code), normalized_hex_id)
    direct_key = (int(fbp_code), None)

    if hex_key in curve_lookup:
        fuel_vector = curve_lookup[hex_key]
    elif direct_key in curve_lookup:
        fuel_vector = curve_lookup[direct_key]
    else:
        raise KeyError(f"No vector found for fbp_code={int(fbp_code)}, hex_id={normalized_hex_id}")

    if copy:
        return fuel_vector.copy()

    return fuel_vector


def compute_fuel_curve_norm_stats(
    root_dir: str | Path,
    raw_data_dir: str | Path,
    feature_name: str,
    allowed_hex_ids: set[int] | None = None,
    fuel_curves_filename: str = DEFAULT_FUEL_CURVES_CSV,
) -> tuple[float, float]:
    """Compute log1p mean and std of fuel curve vectors for offline caching.

    Parameters
    ----------
    root_dir
        Prepared patch dataset directory (contains fuel curve CSV and ignition
        distribution tables used by ``build_fuel_curve_lookup``).
    raw_data_dir
        Raw per-hex raster directory (used by ``build_fuel_curve_lookup`` to
        locate per-hex ignition distribution files).
    feature_name
        Curve feature key, e.g. ``"iROS"`` or ``"HFI"``.
    allowed_hex_ids
        If provided, only hex-specific vectors from these hexels contribute to
        the stats (hex-independent vectors always contribute).
    fuel_curves_filename
        Filename (relative to root_dir) of the fuel curve CSV. Defaults to
        ``DEFAULT_FUEL_CURVES_CSV``.

    Returns
    -------
    (mean, std) of log1p-transformed curve values.
    """
    # Assumption: normalization stats use all curve values since all classes are present in training data for the national study.
    lookup = build_fuel_curve_lookup(
        root_dir=root_dir,
        raw_data_dir=raw_data_dir,
        feature_name=feature_name,
        fuel_curves_filename=fuel_curves_filename,
    )

    if allowed_hex_ids is not None:
        train_hex_strs = {str(hid).zfill(2) for hid in allowed_hex_ids}
        vectors = [vec for (_, hex_id), vec in lookup.items() if hex_id is None or hex_id in train_hex_strs]
    else:
        vectors = list(lookup.values())

    if not vectors:
        raise ValueError(
            f"No fuel curve vectors found for feature_name={feature_name!r}. "
            "Check root_dir and raw_data_dir point to the correct dataset."
        )

    log_vecs = np.log1p(np.clip(np.stack(vectors, axis=0), 0, None))
    return float(log_vecs.mean()), float(log_vecs.std())

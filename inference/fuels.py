"""
Fuel curves for the fuel codes of a project, including codes the model's fuel table does not list.

The model never sees fuel codes: each code is turned into its FBP initial rate of spread curve
(ROS as a function of ISI, see ``data_preparation/tabular/fuel_features``). A regional project can use
codes the national table lacks, e.g. mixedwood with another percent conifer (``M-1 (25 PC)``). BurnP3+
projects say what every code is in ``hexNN_FuelTypes.csv`` (name -> raster ID) and
``hexNN_FuelCodeCrosswalk.csv`` (name -> FBP code), so the curve of such a code is computed here with the
same FBP equations and settings as ``compute_vector_values_national.R`` (ISI given directly, BUI effect off,
BUI 60, grass curing 90%). The equations reproduce every curve of the national and NWT tables (to 1e-13).

Only fuel types the model was trained on and whose curves were verified against R are derived. Others
(C-6, M-3/M-4, S-1 to S-3) must be generated with the R script.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from data_preparation.paths import Paths

CODE_COL = "fbp_code"
SEASON_COL = "SeasonState"
ISI_COL = "ISI"
ROS_COL = "ROS"
LABEL_COL = "CurveLabel"
FUEL_TYPE_COL = "FuelType"

DIRECT, LEAFLESS, GREEN, NONFUEL = "direct", "leafless", "green", "nonfuel"

# FBP initial spread rate RSI = a * (1 - exp(-b * ISI)) ** c (Forestry Canada 1992; Wotton et al. 2009).
_RSI_PARAMS: dict[str, tuple[float, float, float]] = {
    "C-1": (90.0, 0.0649, 4.5),
    "C-2": (110.0, 0.0282, 1.5),
    "C-3": (110.0, 0.0444, 3.0),
    "C-4": (110.0, 0.0293, 1.5),
    "C-5": (30.0, 0.0697, 4.0),
    "C-7": (45.0, 0.0305, 2.0),
    "D-1": (30.0, 0.0232, 1.6),
    "O-1a": (190.0, 0.0310, 1.4),
    "O-1b": (250.0, 0.0350, 1.7),
}
# Grass curing (%) used for the O-1 curves of the national table.
GRASS_CURING = 90.0
# cffdrs::fbp returns this instead of a non-positive spread rate.
_ROS_FLOOR = 1e-6
# Fuel types whose curve needs the R script (not verified here, and absent from the national training data).
UNSUPPORTED_FUEL_TYPES = frozenset({"C-6", "M-3", "M-4", "S-1", "S-2", "S-3"})
_SEASONAL = {"D-1": LEAFLESS, "D-2": GREEN, "M-1": LEAFLESS, "M-2": GREEN, "M-3": LEAFLESS, "M-4": GREEN}
_NONFUEL_CODES = {"non-fuel", "nonfuel", "nf", "wa", "water"}
_CODE_PATTERN = re.compile(
    r"^(?P<types>[A-Z]-\d[a-z]?(?:\s*/\s*[A-Z]-\d[a-z]?)?)\s*(?:\(\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<kind>PC|PDF)\s*\))?$"
)


class FuelDefinitionError(ValueError):
    """A fuel code whose curve cannot be derived; the message says why."""


@dataclass(frozen=True)
class FuelComponent:
    """One FBP fuel type used in one season state of a fuel code."""

    season_state: str
    fuel_type: str
    percent_conifer: float | None = None


def parse_crosswalk_code(code: str) -> tuple[FuelComponent, ...]:
    """Components of a FuelCodeCrosswalk FBP code, e.g. ``M-1/M-2 (80 PC)`` -> leafless M-1 and green M-2 at 80 PC."""
    text = " ".join(str(code).split())
    if text.lower() in _NONFUEL_CODES:
        return (FuelComponent(NONFUEL, "NF"),)
    match = _CODE_PATTERN.match(text)
    if match is None:
        raise FuelDefinitionError(f"unrecognised FBP code {text!r}")
    types = [part.strip() for part in match.group("types").split("/")]
    value = float(match.group("value")) if match.group("value") else None
    kind = match.group("kind")
    unsupported = [fuel_type for fuel_type in types if fuel_type in UNSUPPORTED_FUEL_TYPES]
    if unsupported:
        raise FuelDefinitionError(f"{text}: fuel type {'/'.join(unsupported)} needs curves from the R script")
    unknown = [fuel_type for fuel_type in types if fuel_type not in _RSI_PARAMS and fuel_type not in _SEASONAL]
    if unknown:
        raise FuelDefinitionError(f"{text}: unknown FBP fuel type {'/'.join(unknown)}")
    if len(types) == 2 and [_SEASONAL.get(t) for t in types] != [LEAFLESS, GREEN]:
        raise FuelDefinitionError(f"{text}: expected a leafless/green pair such as M-1/M-2 or D-1/D-2")
    mixedwood = any(fuel_type in ("M-1", "M-2") for fuel_type in types)
    if mixedwood:
        if kind != "PC" or value is None:
            raise FuelDefinitionError(f"{text}: mixedwood needs a percent conifer, e.g. {types[0]} (25 PC)")
        if not 0.0 <= value <= 100.0:
            raise FuelDefinitionError(f"{text}: percent conifer must be between 0 and 100")
    elif kind is not None:
        raise FuelDefinitionError(f"{text}: {kind} only applies to mixedwood fuel types")
    return tuple(FuelComponent(_SEASONAL.get(fuel_type, DIRECT), fuel_type, value if mixedwood else None) for fuel_type in types)


def _rsi(fuel_type: str, isi: np.ndarray) -> np.ndarray:
    a, b, c = _RSI_PARAMS[fuel_type]
    return a * (1.0 - np.exp(-b * isi)) ** c


def _grass_curing_factor(curing: float) -> float:
    return 0.005 * (np.exp(0.061 * curing) - 1.0) if curing < 58.8 else 0.176 + 0.02 * (curing - 58.8)


def fbp_ros_curve(component: FuelComponent, isi: np.ndarray) -> np.ndarray:
    """Initial rate of spread (m/min) of one fuel component at each ISI, as in the national fuel table."""
    isi = np.asarray(isi, dtype=float)
    fuel_type = component.fuel_type
    if fuel_type == "NF":
        return np.zeros_like(isi)
    if fuel_type in ("M-1", "M-2"):
        conifer = (component.percent_conifer or 0.0) / 100.0
        # M-2 (green): the deciduous part spreads at 20% of D-1.
        deciduous = 1.0 if fuel_type == "M-1" else 0.2
        ros = conifer * _rsi("C-2", isi) + deciduous * (1.0 - conifer) * _rsi("D-1", isi)
    elif fuel_type == "D-2":
        # D-2 only spreads above BUI 80; the national curves use BUI 60.
        ros = np.zeros_like(isi)
    elif fuel_type in ("O-1a", "O-1b"):
        ros = _rsi(fuel_type, isi) * _grass_curing_factor(GRASS_CURING)
    else:
        ros = _rsi(fuel_type, isi)
    return np.where(ros <= 0.0, _ROS_FLOOR, ros)


def read_project_fuel_codes(paths: Paths, hex_id: str) -> dict[int, tuple[str, str | None]] | None:
    """``{raster ID: (fuel name, FBP code)}`` from hexNN_FuelTypes.csv and hexNN_FuelCodeCrosswalk.csv.

    Returns None when either table is missing. The FBP code is None when the crosswalk has no entry.
    """
    types_path, crosswalk_path = paths.fuel_table(hex_id), paths.fuel_crosswalk_table(hex_id)
    if not types_path.is_file() or not crosswalk_path.is_file():
        return None
    fuel_types = pd.read_csv(types_path)
    crosswalk = pd.read_csv(crosswalk_path)
    for table, columns, path in ((fuel_types, ("Name", "ID"), types_path), (crosswalk, ("FuelType", "Code"), crosswalk_path)):
        missing = [column for column in columns if column not in table.columns]
        if missing:
            raise ValueError(f"{path} is missing column(s) {missing}.")
    fbp_codes = {str(name).strip(): code for name, code in zip(crosswalk["FuelType"], crosswalk["Code"], strict=True)}
    codes: dict[int, tuple[str, str | None]] = {}
    for name, raster_id in zip(fuel_types["Name"], fuel_types["ID"], strict=True):
        if pd.isna(raster_id):
            continue
        fbp_code = fbp_codes.get(str(name).strip())
        codes[int(raster_id)] = (str(name).strip(), None if fbp_code is None or pd.isna(fbp_code) else str(fbp_code).strip())
    return codes


@dataclass
class FuelCurveResolution:
    """The fuel curve table to use for a project, and how codes missing from the model's table were handled."""

    curves: pd.DataFrame
    derived: dict[int, str] = field(default_factory=dict)  # code -> FBP code, curve computed from the project tables
    replaced: dict[int, str] = field(default_factory=dict)  # code the project defines differently from the model
    unresolved: dict[int, str] = field(default_factory=dict)  # code -> reason no curve is available
    unverified: dict[int, str] = field(default_factory=dict)  # known code whose project definition cannot be derived -> reason

    @property
    def changed(self) -> bool:
        return bool(self.derived or self.replaced)

    def summary(self) -> dict[str, dict[str, str]]:
        return {
            "derived": {str(code): value for code, value in sorted(self.derived.items())},
            "replaced": {str(code): value for code, value in sorted(self.replaced.items())},
        }


def _code_rows(code: int, label: str, components: tuple[FuelComponent, ...], isi: np.ndarray) -> pd.DataFrame:
    return pd.concat(
        [
            pd.DataFrame(
                {
                    CODE_COL: code,
                    LABEL_COL: f"{label} {component.season_state}" if len(components) > 1 else label,
                    FUEL_TYPE_COL: component.fuel_type,
                    SEASON_COL: component.season_state,
                    ISI_COL: isi,
                    ROS_COL: fbp_ros_curve(component, isi),
                }
            )
            for component in components
        ],
        ignore_index=True,
    )


def _same_curves(model_rows: pd.DataFrame, project_rows: pd.DataFrame) -> bool:
    def by_state(rows: pd.DataFrame) -> dict[str, np.ndarray]:
        return {str(state): group.sort_values(ISI_COL)[ROS_COL].to_numpy(float) for state, group in rows.groupby(SEASON_COL)}

    model, project = by_state(model_rows), by_state(project_rows)
    if len(model) == 1 and len(project) == 1:  # single-state codes: the state name (direct/nonfuel/water) does not matter
        model, project = {"": next(iter(model.values()))}, {"": next(iter(project.values()))}
    if model.keys() != project.keys():
        return False
    return all(np.allclose(model[state], project[state], rtol=1e-6, atol=1e-9) for state in model)


def resolve_fuel_curves(
    model_curves: pd.DataFrame,
    project_codes: Mapping[int, tuple[str, str | None]] | None,
    codes: Iterable[int] | None = None,
    feature_col: str = ROS_COL,
    project_tables: str = "hexNN_FuelTypes.csv / hexNN_FuelCodeCrosswalk.csv",
) -> FuelCurveResolution:
    """Model curves plus curves derived from the project's fuel tables for ``codes`` (default: every project code).

    Codes the model's table lists keep the model's curve unless the project defines them as another fuel,
    in which case the project's definition is used (``replaced``); when that definition cannot be derived, the
    model's curve is kept (``unverified``).
    """
    known = {int(code) for code in model_curves[CODE_COL].unique()}
    wanted = sorted({int(code) for code in (codes if codes is not None else (project_codes or {}))})
    isi = np.sort(model_curves[ISI_COL].unique().astype(float))
    trained_types = set(model_curves[FUEL_TYPE_COL].astype(str)) if FUEL_TYPE_COL in model_curves else None
    resolution = FuelCurveResolution(curves=model_curves)
    added: list[pd.DataFrame] = []
    replaced_codes: set[int] = set()
    for code in wanted:
        entry = (project_codes or {}).get(code)
        if feature_col != ROS_COL:
            if code not in known:
                resolution.unresolved[code] = f"not in the model's fuel table; {feature_col} curves must come from the R script"
            continue
        if entry is None:
            if code not in known:
                where = f"not listed in {project_tables}" if project_codes is not None else f"no {project_tables} to define it"
                resolution.unresolved[code] = f"not in the model's fuel table and {where}"
            continue
        name, fbp_code = entry
        if fbp_code is None:
            if code not in known:
                resolution.unresolved[code] = f"{name!r} has no FBP code in the crosswalk"
            continue
        try:
            components = parse_crosswalk_code(fbp_code)
            untrained = sorted({c.fuel_type for c in components if c.fuel_type != "NF"} - (trained_types or set()))
            if trained_types is not None and untrained:
                raise FuelDefinitionError(f"{fbp_code}: fuel type {'/'.join(untrained)} is not in the model's training data")
        except FuelDefinitionError as exc:
            (resolution.unresolved if code not in known else resolution.unverified)[code] = str(exc)
            continue
        rows = _code_rows(code, fbp_code, components, isi)
        if code not in known:
            resolution.derived[code] = fbp_code
            added.append(rows)
        elif not _same_curves(model_curves[model_curves[CODE_COL] == code], rows):
            resolution.replaced[code] = fbp_code
            replaced_codes.add(code)
            added.append(rows)
    if added:
        kept = model_curves[~model_curves[CODE_COL].isin(replaced_codes)]
        resolution.curves = pd.concat([kept, *added], ignore_index=True)
    return resolution


def read_fuel_curves(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def describe_codes(codes: Mapping[int, str], counts: Mapping[int, int] | None = None, limit: int = 12) -> str:
    items = [
        f"{code} = {value}" + (f" ({counts[code]:,} cells)" if counts and code in counts else "") for code, value in sorted(codes.items())
    ]
    shown = ", ".join(items[:limit])
    return shown + (f" and {len(items) - limit} more" if len(items) > limit else "")

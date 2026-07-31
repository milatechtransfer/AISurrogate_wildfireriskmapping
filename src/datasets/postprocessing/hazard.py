"""Array-level wildfire hazard math.

The functions here avoid file paths and model config objects so the same rules
can be reused by config validation, tests, and evaluation pipelines.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

DEFAULT_FI_CAP = 10000.0
DEFAULT_SCALE_TO = 100.0
DEFAULT_HAZARD_BIN_THRESHOLDS = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0]


def _validate_positive(value: float, name: str) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return numeric


def validate_bin_thresholds(thresholds: Sequence[float], name: str = "thresholds") -> np.ndarray:
    arr = np.asarray(thresholds, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D sequence")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must all be finite")
    if np.any(arr <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return arr


def cap_fire_intensity(fi_grid: np.ndarray, fi_cap: float | None = DEFAULT_FI_CAP) -> np.ndarray:
    """Cap fire intensity at ``fi_cap`` pixelwise, preserving NaNs. ``None`` leaves it uncapped."""
    fi = np.asarray(fi_grid, dtype=float)
    if fi_cap is None:
        return fi.copy()
    cap = _validate_positive(fi_cap, "fi_cap")
    return np.minimum(fi, cap)


def compute_raw_hazard(
    bp_grid: np.ndarray,
    fi_grid: np.ndarray,
    fi_cap: float | None = DEFAULT_FI_CAP,
) -> np.ndarray:
    """Compute ``raw_hazard = BP * min(FI, fi_cap)`` pixelwise.

    ``bp_grid`` and ``fi_grid`` must share an identical shape. The result is a
    float array; NaNs in either input propagate to the output.
    """
    bp = np.asarray(bp_grid, dtype=float)
    fi = np.asarray(fi_grid, dtype=float)
    if bp.shape != fi.shape:
        raise ValueError(f"BP and FI grids must have identical shape, got {bp.shape} and {fi.shape}")
    capped_fi = cap_fire_intensity(fi, fi_cap)
    return bp * capped_fi


def scale_hazard(
    raw_hazard: np.ndarray,
    denominator: float,
    scale_to: float = DEFAULT_SCALE_TO,
) -> np.ndarray:
    """Scale a raw hazard grid: ``scaled = raw_hazard * (scale_to / denominator)``."""
    raw = np.asarray(raw_hazard, dtype=float)
    denom = _validate_positive(denominator, "denominator")
    target = _validate_positive(scale_to, "scale_to")
    return raw * (target / denom)


def bin_scaled_hazard(
    scaled_hazard: np.ndarray,
    thresholds: Sequence[float] | None = None,
    invalid_class: int = 0,
) -> np.ndarray:
    """Bin a scaled hazard grid into 1-based hazard classes.

    Values equal to a threshold move into the higher class; non-finite pixels
    are assigned ``invalid_class``.
    """
    if thresholds is None:
        thresholds = DEFAULT_HAZARD_BIN_THRESHOLDS
    edges = validate_bin_thresholds(thresholds)

    scaled = np.asarray(scaled_hazard, dtype=float)
    classes = np.searchsorted(edges, scaled, side="right") + 1
    classes = classes.astype(np.int64)

    finite = np.isfinite(scaled)
    classes[~finite] = invalid_class
    return classes


def max_finite_hazard(*raw_hazard_grids: np.ndarray) -> float:
    """Return the maximum finite, positive value across one or more raw grids (a denominator)."""
    if not raw_hazard_grids:
        raise ValueError("max_finite_hazard requires at least one grid")

    current_max = -np.inf
    for grid in raw_hazard_grids:
        arr = np.asarray(grid, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            current_max = max(current_max, float(finite.max()))

    if not np.isfinite(current_max):
        raise ValueError("no finite hazard values found to compute a denominator")
    if current_max <= 0.0:
        raise ValueError(f"maximum finite hazard must be > 0, got {current_max}")
    return current_max

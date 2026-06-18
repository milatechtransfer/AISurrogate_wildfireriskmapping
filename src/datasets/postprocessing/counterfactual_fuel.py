"""Raw fuel-grid editing helpers for counterfactual analyses."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, find_objects, label

FUEL_NODATA = -32768


@dataclass(frozen=True)
class FuelEditReport:
    """Summary of one fuel counterfactual edit."""

    scenario_name: str
    mode: str
    edited_pixels: int
    replacement_fuel_id: int | None
    candidate_pixels: int
    original_nonfuel_pixels: int
    original_burnable_pixels: int
    note: str

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(self)])


def nonfuel_mask(fuel: np.ndarray, nonfuel_ids: list[int] | tuple[int, ...], nodata_value: int = FUEL_NODATA) -> np.ndarray:
    """Return True where fuel is a valid non-fuel/spread-barrier ID."""

    fuel_arr = np.asarray(fuel)
    finite = np.isfinite(fuel_arr)
    return finite & (fuel_arr != nodata_value) & np.isin(fuel_arr, list(nonfuel_ids))


def burnable_mask(fuel: np.ndarray, nonfuel_ids: list[int] | tuple[int, ...], nodata_value: int = FUEL_NODATA) -> np.ndarray:
    """Return True where fuel is valid and not a non-fuel/spread-barrier ID."""

    fuel_arr = np.asarray(fuel)
    finite = np.isfinite(fuel_arr)
    return finite & (fuel_arr != nodata_value) & ~np.isin(fuel_arr, list(nonfuel_ids))


def _modal_int(values: np.ndarray) -> int:
    if values.size == 0:
        raise ValueError("Cannot compute modal fuel from an empty array.")
    unique, counts = np.unique(values.astype(np.int64), return_counts=True)
    return int(unique[np.argmax(counts)])


def modal_adjacent_burnable_fuel(
    fuel: np.ndarray,
    edit_mask: np.ndarray,
    burnable: np.ndarray,
    *,
    fallback_to_global: bool = True,
) -> tuple[int, int, str]:
    """Pick the modal burnable fuel touching the edit mask.

    Uses an 8-neighbour dilation.  If no adjacent burnable candidate exists and
    ``fallback_to_global`` is true, falls back to the modal burnable fuel in the
    full array and records that in the note.
    """

    if fuel.shape != edit_mask.shape or fuel.shape != burnable.shape:
        raise ValueError("fuel, edit_mask, and burnable must have the same shape.")

    structure = np.ones((3, 3), dtype=bool)
    adjacent = binary_dilation(edit_mask, structure=structure) & ~edit_mask & burnable
    candidate_values = fuel[adjacent]
    if candidate_values.size > 0:
        return _modal_int(candidate_values), int(candidate_values.size), "modal adjacent burnable fuel"

    if not fallback_to_global:
        raise ValueError("No adjacent burnable fuel candidates found.")
    global_values = fuel[burnable]
    if global_values.size == 0:
        raise ValueError("No burnable fuel candidates found.")
    return _modal_int(global_values), int(global_values.size), "fallback modal global burnable fuel"


def modal_adjacent_burnable_fuel_across_grids(
    fuel_grids: Iterable[np.ndarray],
    nonfuel_ids: list[int] | tuple[int, ...],
    *,
    nodata_value: int = FUEL_NODATA,
) -> tuple[int, int, str]:
    """Pick one replacement fuel ID from adjacent burnable support across grids.

    This is useful for patch-based counterfactual datasets: overlapping patches
    should use a single replacement fuel so the same geographic pixel is not
    represented with different fuel codes in different windows.
    """

    adjacent_counts: Counter[int] = Counter()
    global_burnable_counts: Counter[int] = Counter()
    structure = np.ones((3, 3), dtype=bool)
    seen_any_grid = False

    for fuel in fuel_grids:
        seen_any_grid = True
        fuel_arr = np.asarray(fuel)
        base_nonfuel = nonfuel_mask(fuel_arr, nonfuel_ids, nodata_value=nodata_value)
        base_burnable = burnable_mask(fuel_arr, nonfuel_ids, nodata_value=nodata_value)

        adjacent = binary_dilation(base_nonfuel, structure=structure) & ~base_nonfuel & base_burnable
        adjacent_counts.update(map(int, fuel_arr[adjacent].ravel()))
        global_burnable_counts.update(map(int, fuel_arr[base_burnable].ravel()))

    if not seen_any_grid:
        raise ValueError("Cannot choose a replacement fuel from an empty grid iterable.")

    if adjacent_counts:
        replacement, count = max(adjacent_counts.items(), key=lambda item: (item[1], -item[0]))
        return int(replacement), int(sum(adjacent_counts.values())), "modal adjacent burnable fuel across grids"

    if global_burnable_counts:
        replacement, count = max(global_burnable_counts.items(), key=lambda item: (item[1], -item[0]))
        return int(replacement), int(sum(global_burnable_counts.values())), "fallback modal global burnable fuel across grids"

    raise ValueError("No burnable fuel candidates found across grids.")


def replace_nonfuel_with_burnable(
    fuel: np.ndarray,
    nonfuel_ids: list[int] | tuple[int, ...],
    replacement_fuel_id: int,
    *,
    scenario_name: str = "barrier_removal_fixed_burnable",
    edit_mask: np.ndarray | None = None,
    nodata_value: int = FUEL_NODATA,
    candidate_pixels: int = 0,
    note: str = "selected non-fuel pixels replaced with fixed burnable fuel",
) -> tuple[np.ndarray, np.ndarray, FuelEditReport]:
    """Replace selected non-fuel pixels with a fixed burnable fuel ID."""

    if replacement_fuel_id in set(nonfuel_ids):
        raise ValueError(f"replacement_fuel_id={replacement_fuel_id} is a non-fuel ID.")

    fuel_arr = np.asarray(fuel)
    base_nonfuel = nonfuel_mask(fuel_arr, nonfuel_ids, nodata_value=nodata_value)
    base_burnable = burnable_mask(fuel_arr, nonfuel_ids, nodata_value=nodata_value)
    selected = base_nonfuel if edit_mask is None else (np.asarray(edit_mask, dtype=bool) & base_nonfuel)

    edited = fuel_arr.copy()
    edited[selected] = int(replacement_fuel_id)
    report = FuelEditReport(
        scenario_name=scenario_name,
        mode="nonfuel_to_burnable_fixed",
        edited_pixels=int(selected.sum()),
        replacement_fuel_id=int(replacement_fuel_id),
        candidate_pixels=int(candidate_pixels),
        original_nonfuel_pixels=int(base_nonfuel.sum()),
        original_burnable_pixels=int(base_burnable.sum()),
        note=note,
    )
    return edited, selected, report


def replace_nonfuel_with_adjacent_modal(
    fuel: np.ndarray,
    nonfuel_ids: list[int] | tuple[int, ...],
    *,
    scenario_name: str = "barrier_removal_adjacent_modal",
    edit_mask: np.ndarray | None = None,
    nodata_value: int = FUEL_NODATA,
) -> tuple[np.ndarray, np.ndarray, FuelEditReport]:
    """Replace selected non-fuel pixels with the modal adjacent burnable fuel."""

    fuel_arr = np.asarray(fuel)
    base_nonfuel = nonfuel_mask(fuel_arr, nonfuel_ids, nodata_value=nodata_value)
    base_burnable = burnable_mask(fuel_arr, nonfuel_ids, nodata_value=nodata_value)
    selected = base_nonfuel if edit_mask is None else (np.asarray(edit_mask, dtype=bool) & base_nonfuel)

    edited = fuel_arr.copy()
    if not selected.any():
        report = FuelEditReport(
            scenario_name=scenario_name,
            mode="nonfuel_to_burnable_adjacent_modal",
            edited_pixels=0,
            replacement_fuel_id=None,
            candidate_pixels=0,
            original_nonfuel_pixels=int(base_nonfuel.sum()),
            original_burnable_pixels=int(base_burnable.sum()),
            note="no selected non-fuel pixels",
        )
        return edited, selected, report

    replacement, candidate_pixels, note = modal_adjacent_burnable_fuel(fuel_arr, selected, base_burnable)
    edited[selected] = replacement
    report = FuelEditReport(
        scenario_name=scenario_name,
        mode="nonfuel_to_burnable_adjacent_modal",
        edited_pixels=int(selected.sum()),
        replacement_fuel_id=replacement,
        candidate_pixels=candidate_pixels,
        original_nonfuel_pixels=int(base_nonfuel.sum()),
        original_burnable_pixels=int(base_burnable.sum()),
        note=note,
    )
    return edited, selected, report


def replace_nonfuel_components_with_adjacent_modal(
    fuel: np.ndarray,
    nonfuel_ids: list[int] | tuple[int, ...],
    *,
    scenario_name: str = "barrier_removal_local_adjacent_modal",
    edit_mask: np.ndarray | None = None,
    nodata_value: int = FUEL_NODATA,
) -> tuple[np.ndarray, np.ndarray, FuelEditReport, pd.DataFrame]:
    """Replace each connected non-fuel component with its local adjacent modal fuel.

    Components use 8-neighbour connectivity.  For each component, the
    replacement is the modal burnable fuel among pixels touching that component.
    This keeps the counterfactual local while avoiding inconsistent replacements
    in overlapping patch windows.
    """

    fuel_arr = np.asarray(fuel)
    base_nonfuel = nonfuel_mask(fuel_arr, nonfuel_ids, nodata_value=nodata_value)
    base_burnable = burnable_mask(fuel_arr, nonfuel_ids, nodata_value=nodata_value)
    selected = base_nonfuel if edit_mask is None else (np.asarray(edit_mask, dtype=bool) & base_nonfuel)
    edited = fuel_arr.copy()

    if not selected.any():
        report = FuelEditReport(
            scenario_name=scenario_name,
            mode="nonfuel_to_burnable_local_adjacent_modal",
            edited_pixels=0,
            replacement_fuel_id=None,
            candidate_pixels=0,
            original_nonfuel_pixels=int(base_nonfuel.sum()),
            original_burnable_pixels=int(base_burnable.sum()),
            note="no selected non-fuel pixels",
        )
        return edited, selected, report, pd.DataFrame()

    structure = np.ones((3, 3), dtype=np.int8)
    component_labels, n_components = label(selected, structure=structure)
    global_values = fuel_arr[base_burnable]
    if global_values.size == 0:
        raise ValueError("No burnable fuel candidates found.")

    rows: list[dict] = []
    replacement_counts: Counter[int] = Counter()
    total_candidate_pixels = 0
    component_slices = find_objects(component_labels)
    for component_id in range(1, int(n_components) + 1):
        component_slice = component_slices[component_id - 1]
        if component_slice is None:
            continue
        row_slice, col_slice = component_slice
        row_window = slice(max(0, row_slice.start - 1), min(fuel_arr.shape[0], row_slice.stop + 1))
        col_window = slice(max(0, col_slice.start - 1), min(fuel_arr.shape[1], col_slice.stop + 1))
        sub_labels = component_labels[row_window, col_window]
        component = sub_labels == component_id
        adjacent = binary_dilation(component, structure=structure.astype(bool)) & ~component & base_burnable[row_window, col_window]
        candidate_values = fuel_arr[row_window, col_window][adjacent]
        if candidate_values.size > 0:
            replacement = _modal_int(candidate_values)
            note = "modal adjacent burnable fuel for connected component"
            candidate_pixels = int(candidate_values.size)
        else:
            replacement = _modal_int(global_values)
            note = "fallback modal global burnable fuel for connected component"
            candidate_pixels = int(global_values.size)

        edited_window = edited[row_window, col_window]
        edited_window[component] = replacement
        edited_pixels = int(component.sum())
        replacement_counts[replacement] += edited_pixels
        total_candidate_pixels += candidate_pixels
        rows.append(
            {
                "component_id": component_id,
                "edited_pixels": edited_pixels,
                "replacement_fuel_id": replacement,
                "candidate_pixels": candidate_pixels,
                "note": note,
            }
        )

    dominant_replacement = replacement_counts.most_common(1)[0][0]
    replacement_ids = ";".join(map(str, sorted(replacement_counts)))
    report = FuelEditReport(
        scenario_name=scenario_name,
        mode="nonfuel_to_burnable_local_adjacent_modal",
        edited_pixels=int(selected.sum()),
        replacement_fuel_id=int(dominant_replacement),
        candidate_pixels=int(total_candidate_pixels),
        original_nonfuel_pixels=int(base_nonfuel.sum()),
        original_burnable_pixels=int(base_burnable.sum()),
        note=f"{int(n_components)} connected components; replacement_fuel_ids={replacement_ids}",
    )
    return edited, selected, report, pd.DataFrame(rows)


def replace_burnable_with_nonfuel(
    fuel: np.ndarray,
    nonfuel_ids: list[int] | tuple[int, ...],
    insertion_mask: np.ndarray,
    *,
    replacement_nonfuel_id: int | None = None,
    scenario_name: str = "barrier_insertion",
    nodata_value: int = FUEL_NODATA,
) -> tuple[np.ndarray, np.ndarray, FuelEditReport]:
    """Replace selected burnable pixels with a non-fuel ID."""

    if not nonfuel_ids:
        raise ValueError("At least one non-fuel ID is required for barrier insertion.")
    replacement = int(replacement_nonfuel_id if replacement_nonfuel_id is not None else sorted(nonfuel_ids)[0])
    if replacement not in set(nonfuel_ids):
        raise ValueError(f"replacement_nonfuel_id={replacement} is not in nonfuel_ids={sorted(nonfuel_ids)}.")

    fuel_arr = np.asarray(fuel)
    base_nonfuel = nonfuel_mask(fuel_arr, nonfuel_ids, nodata_value=nodata_value)
    base_burnable = burnable_mask(fuel_arr, nonfuel_ids, nodata_value=nodata_value)
    selected = np.asarray(insertion_mask, dtype=bool) & base_burnable

    edited = fuel_arr.copy()
    edited[selected] = replacement
    report = FuelEditReport(
        scenario_name=scenario_name,
        mode="burnable_to_nonfuel",
        edited_pixels=int(selected.sum()),
        replacement_fuel_id=replacement,
        candidate_pixels=int(base_burnable.sum()),
        original_nonfuel_pixels=int(base_nonfuel.sum()),
        original_burnable_pixels=int(base_burnable.sum()),
        note="selected burnable pixels replaced with non-fuel",
    )
    return edited, selected, report

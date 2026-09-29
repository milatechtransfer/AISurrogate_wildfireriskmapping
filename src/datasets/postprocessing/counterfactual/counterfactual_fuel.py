"""Raw fuel-grid editing helpers for counterfactual analyses."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, find_objects, label

FUEL_NODATA = -32768
FUEL_EDIT_MODES = (
    "nonfuel_to_burnable_fixed",
    "nonfuel_to_burnable_adjacent_modal",
    "nonfuel_to_burnable_local_adjacent_modal",
    "burnable_to_nonfuel",
    "burnable_components_to_nonfuel_random",
    "burnable_to_burnable_fixed",
)


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


@dataclass(frozen=True)
class FuelEditResult:
    fuel: np.ndarray
    edit_mask: np.ndarray
    report: FuelEditReport
    components: pd.DataFrame


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


def replace_burnable_with_fixed(
    fuel: np.ndarray,
    nonfuel_ids: list[int] | tuple[int, ...],
    source_fuel_ids: list[int] | tuple[int, ...] | None,
    replacement_fuel_id: int,
    *,
    scenario_name: str = "fuel_type_substitution",
    edit_mask: np.ndarray | None = None,
    nodata_value: int = FUEL_NODATA,
) -> tuple[np.ndarray, np.ndarray, FuelEditReport]:
    """Replace burnable pixels with a fixed burnable fuel ID.

    `source_fuel_ids=None` targets every burnable pixel, which is the useful
    default when `edit_mask` already restricts the edit to a region (e.g. fire
    perimeters). An empty list is rejected as a likely config mistake.
    """

    if source_fuel_ids is not None and not source_fuel_ids:
        raise ValueError("source_fuel_ids must be omitted (to target all burnable fuel) or contain at least one ID.")
    if source_fuel_ids is not None and set(source_fuel_ids) & set(nonfuel_ids):
        raise ValueError(f"source_fuel_ids={sorted(source_fuel_ids)} must not overlap nonfuel_ids={sorted(nonfuel_ids)}.")
    if replacement_fuel_id in set(nonfuel_ids):
        raise ValueError(f"replacement_fuel_id={replacement_fuel_id} is a non-fuel ID.")

    fuel_arr = np.asarray(fuel)
    base_nonfuel = nonfuel_mask(fuel_arr, nonfuel_ids, nodata_value=nodata_value)
    base_burnable = burnable_mask(fuel_arr, nonfuel_ids, nodata_value=nodata_value)
    if source_fuel_ids is None:
        base_source = base_burnable
        note = "all burnable fuel replaced with fixed burnable fuel"
    else:
        base_source = base_burnable & np.isin(fuel_arr, list(source_fuel_ids))
        note = f"source_fuel_ids={sorted(source_fuel_ids)} replaced with fixed burnable fuel"
    selected = base_source if edit_mask is None else (np.asarray(edit_mask, dtype=bool) & base_source)

    edited = fuel_arr.copy()
    edited[selected] = int(replacement_fuel_id)
    report = FuelEditReport(
        scenario_name=scenario_name,
        mode="burnable_to_burnable_fixed",
        edited_pixels=int(selected.sum()),
        replacement_fuel_id=int(replacement_fuel_id),
        candidate_pixels=int(base_source.sum()),
        original_nonfuel_pixels=int(base_nonfuel.sum()),
        original_burnable_pixels=int(base_burnable.sum()),
        note=note,
    )
    return edited, selected, report


def replace_random_burnable_components_with_nonfuel(
    fuel: np.ndarray,
    nonfuel_ids: list[int] | tuple[int, ...],
    *,
    replacement_nonfuel_id: int,
    target_burnable_area_fraction: float,
    seed: int,
    scenario_name: str = "random_barrier_insertion",
    nodata_value: int = FUEL_NODATA,
) -> tuple[np.ndarray, np.ndarray, FuelEditReport, pd.DataFrame]:
    """Replace area-weighted random same-fuel components up to an area target."""

    if not 0.0 < target_burnable_area_fraction <= 1.0:
        raise ValueError("target_burnable_area_fraction must be in (0, 1].")
    if replacement_nonfuel_id not in set(nonfuel_ids):
        raise ValueError(f"replacement_nonfuel_id={replacement_nonfuel_id} is not in nonfuel_ids={sorted(nonfuel_ids)}.")

    fuel_arr = np.asarray(fuel)
    base_nonfuel = nonfuel_mask(fuel_arr, nonfuel_ids, nodata_value=nodata_value)
    base_burnable = burnable_mask(fuel_arr, nonfuel_ids, nodata_value=nodata_value)
    original_burnable_pixels = int(base_burnable.sum())
    if original_burnable_pixels == 0:
        raise ValueError("No burnable fuel components found.")

    target_pixels = int(np.ceil(original_burnable_pixels * target_burnable_area_fraction))
    rng = np.random.default_rng(seed)
    structure = np.ones((3, 3), dtype=np.int8)
    candidates: list[tuple[int, int, int]] = []
    for fuel_id in np.unique(fuel_arr[base_burnable]).astype(np.int64):
        component_labels, n_components = label(base_burnable & (fuel_arr == fuel_id), structure=structure)
        component_sizes = np.bincount(component_labels.ravel(), minlength=int(n_components) + 1)
        candidates.extend(
            (int(fuel_id), component_id, int(component_sizes[component_id])) for component_id in range(1, int(n_components) + 1)
        )

    weights = np.asarray([component_pixels for _, _, component_pixels in candidates], dtype=np.float64)
    weighted_order = np.argsort(rng.exponential(scale=1.0 / weights))
    selected_candidates: list[tuple[int, int, int]] = []
    selected_pixels = 0
    for candidate_index in weighted_order:
        candidate = candidates[int(candidate_index)]
        selected_candidates.append(candidate)
        selected_pixels += candidate[2]
        if selected_pixels >= target_pixels:
            break

    selected = np.zeros(fuel_arr.shape, dtype=bool)
    rows: list[dict] = []
    selected_by_fuel: dict[int, list[tuple[int, int]]] = {}
    for fuel_id, component_id, component_pixels in selected_candidates:
        selected_by_fuel.setdefault(fuel_id, []).append((component_id, component_pixels))
    for fuel_id, fuel_components in selected_by_fuel.items():
        component_labels, _ = label(base_burnable & (fuel_arr == fuel_id), structure=structure)
        selected_ids = [component_id for component_id, _ in fuel_components]
        selected |= np.isin(component_labels, selected_ids)
        row_offset = len(rows)
        rows.extend(
            {
                "component_id": row_offset + offset + 1,
                "source_component_id": component_id,
                "edited_pixels": component_pixels,
                "original_fuel_id": fuel_id,
                "replacement_fuel_id": replacement_nonfuel_id,
                "seed": seed,
            }
            for offset, (component_id, component_pixels) in enumerate(fuel_components)
        )

    edited = fuel_arr.copy()
    edited[selected] = replacement_nonfuel_id
    edited_pixels = int(selected.sum())
    achieved_fraction = edited_pixels / original_burnable_pixels
    report = FuelEditReport(
        scenario_name=scenario_name,
        mode="burnable_components_to_nonfuel_random",
        edited_pixels=edited_pixels,
        replacement_fuel_id=replacement_nonfuel_id,
        candidate_pixels=original_burnable_pixels,
        original_nonfuel_pixels=int(base_nonfuel.sum()),
        original_burnable_pixels=original_burnable_pixels,
        note=(
            f"{len(rows)} area-weighted same-fuel components; "
            f"target_fraction={target_burnable_area_fraction:.6f}; achieved_fraction={achieved_fraction:.6f}; seed={seed}"
        ),
    )
    return edited, selected, report, pd.DataFrame(rows)


def apply_fuel_edit(
    fuel: np.ndarray,
    nonfuel_ids: list[int] | tuple[int, ...],
    *,
    mode: str,
    scenario_name: str,
    params: dict | None = None,
) -> FuelEditResult:
    """Apply a configured fuel edit using a consistent result type."""

    params = dict(params or {})
    if mode == "nonfuel_to_burnable_local_adjacent_modal":
        edited, edit_mask, report, components = replace_nonfuel_components_with_adjacent_modal(
            fuel,
            nonfuel_ids,
            scenario_name=scenario_name,
            edit_mask=params.get("edit_mask"),
        )
    elif mode == "nonfuel_to_burnable_adjacent_modal":
        edited, edit_mask, report = replace_nonfuel_with_adjacent_modal(
            fuel,
            nonfuel_ids,
            scenario_name=scenario_name,
            edit_mask=params.get("edit_mask"),
        )
        components = pd.DataFrame()
    elif mode == "nonfuel_to_burnable_fixed":
        if "replacement_fuel_id" not in params:
            raise ValueError("nonfuel_to_burnable_fixed requires replacement_fuel_id.")
        edited, edit_mask, report = replace_nonfuel_with_burnable(
            fuel,
            nonfuel_ids,
            int(params["replacement_fuel_id"]),
            scenario_name=scenario_name,
            edit_mask=params.get("edit_mask"),
        )
        components = pd.DataFrame()
    elif mode == "burnable_to_nonfuel":
        if "insertion_mask" not in params:
            raise ValueError("burnable_to_nonfuel requires insertion_mask.")
        edited, edit_mask, report = replace_burnable_with_nonfuel(
            fuel,
            nonfuel_ids,
            np.asarray(params["insertion_mask"], dtype=bool),
            replacement_nonfuel_id=params.get("replacement_nonfuel_id"),
            scenario_name=scenario_name,
        )
        components = pd.DataFrame()
    elif mode == "burnable_to_burnable_fixed":
        if "replacement_fuel_id" not in params:
            raise ValueError("burnable_to_burnable_fixed requires parameter: replacement_fuel_id.")
        raw_source_ids = params.get("source_fuel_ids")
        source_ids = None if raw_source_ids is None else [int(value) for value in raw_source_ids]
        edited, edit_mask, report = replace_burnable_with_fixed(
            fuel,
            nonfuel_ids,
            source_ids,
            int(params["replacement_fuel_id"]),
            scenario_name=scenario_name,
            edit_mask=params.get("edit_mask"),
        )
        components = pd.DataFrame()
    elif mode == "burnable_components_to_nonfuel_random":
        required = {"replacement_nonfuel_id", "target_burnable_area_fraction", "seed"}
        missing = sorted(required - params.keys())
        if missing:
            raise ValueError(f"burnable_components_to_nonfuel_random requires parameters: {missing}.")
        edited, edit_mask, report, components = replace_random_burnable_components_with_nonfuel(
            fuel,
            nonfuel_ids,
            replacement_nonfuel_id=int(params["replacement_nonfuel_id"]),
            target_burnable_area_fraction=float(params["target_burnable_area_fraction"]),
            seed=int(params["seed"]),
            scenario_name=scenario_name,
        )
    else:
        raise ValueError(f"Unknown fuel edit mode {mode!r}; expected one of {FUEL_EDIT_MODES}.")

    return FuelEditResult(fuel=edited, edit_mask=edit_mask, report=report, components=components)

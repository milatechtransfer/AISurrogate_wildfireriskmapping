from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from src.counterfactual.counterfactual_fuel import (
    apply_fuel_edit,
    burnable_mask,
    modal_adjacent_burnable_fuel,
    nonfuel_mask,
    replace_burnable_with_fixed,
    replace_burnable_with_nonfuel,
    replace_nonfuel_components_with_adjacent_modal,
    replace_nonfuel_with_adjacent_modal,
    replace_nonfuel_with_burnable,
    replace_random_burnable_components_with_nonfuel,
)
from src.counterfactual.fuel_counterfactual_transform import fuel_intervention_raster_path
from src.counterfactual.plotting.counterfactual_fuel_intervention_map import (
    intervention_layers,
    intervention_layers_on_prediction_grid,
    summarize_intervention,
)
from src.counterfactual.plotting.counterfactual_viz import zone_boundary_segments


def test_nonfuel_and_burnable_masks_are_complements_on_valid_fuel() -> None:
    fuel = np.array([[101, 1, -32768], [2, 100, 3]], dtype=np.int32)
    nonfuel = nonfuel_mask(fuel, [100, 101])
    burnable = burnable_mask(fuel, [100, 101])
    assert nonfuel.tolist() == [[True, False, False], [False, True, False]]
    assert burnable.tolist() == [[False, True, False], [True, False, True]]


def test_modal_adjacent_burnable_fuel_uses_local_neighbours() -> None:
    fuel = np.array(
        [
            [1, 1, 2],
            [1, 101, 2],
            [3, 3, 2],
        ],
        dtype=np.int32,
    )
    edit = fuel == 101
    burnable = fuel != 101
    replacement, n_candidates, note = modal_adjacent_burnable_fuel(fuel, edit, burnable)
    assert replacement == 1
    assert n_candidates == 8
    assert "adjacent" in note


def test_replace_nonfuel_with_adjacent_modal_reports_edit() -> None:
    fuel = np.array(
        [
            [1, 1, 2],
            [1, 101, 2],
            [3, 3, 2],
        ],
        dtype=np.int32,
    )
    edited, edit_mask, report = replace_nonfuel_with_adjacent_modal(fuel, [101])
    assert edit_mask.sum() == 1
    assert edited[1, 1] == 1
    assert report.edited_pixels == 1
    assert report.replacement_fuel_id == 1


def test_replace_nonfuel_components_with_adjacent_modal_uses_local_components() -> None:
    fuel = np.array(
        [
            [1, 1, 1, 2, 2, 2],
            [1, 0, 1, 2, 0, 2],
            [1, 1, 1, 2, 2, 2],
        ],
        dtype=np.float32,
    )
    edited, edit_mask, report, components = replace_nonfuel_components_with_adjacent_modal(fuel, [0])
    assert edit_mask.sum() == 2
    assert edited[1, 1] == pytest.approx(1.0)
    assert edited[1, 4] == pytest.approx(2.0)
    assert components["replacement_fuel_id"].tolist() == [1, 2]
    assert report.edited_pixels == 2
    assert "2 connected components" in report.note


def test_apply_fuel_edit_returns_consistent_result() -> None:
    fuel = np.array([[1, 1, 2], [1, 0, 2], [1, 1, 2]], dtype=np.float32)

    result = apply_fuel_edit(
        fuel,
        [0],
        mode="nonfuel_to_burnable_local_adjacent_modal",
        scenario_name="remove_barrier",
    )

    assert result.fuel[1, 1] == pytest.approx(1.0)
    assert result.edit_mask.sum() == 1
    assert result.report.scenario_name == "remove_barrier"
    assert result.components["replacement_fuel_id"].tolist() == [1]


def test_intervention_layers_show_only_replaced_nonfuel_pixels() -> None:
    baseline = np.array(
        [
            [1, 0, 2],
            [3, 0, np.nan],
        ],
        dtype=np.float32,
    )
    scenario = np.array(
        [
            [1, 8, 2],
            [3, 14, np.nan],
        ],
        dtype=np.float32,
    )

    original_nonfuel, replacement_map, unexpected = intervention_layers(baseline, scenario)
    assert original_nonfuel.tolist() == [[False, True, False], [False, True, False]]
    assert np.isnan(replacement_map[0, 0])
    assert replacement_map[0, 1] == pytest.approx(8)
    assert replacement_map[1, 1] == pytest.approx(14)
    assert not unexpected.any()

    summary = summarize_intervention(
        scenario="synthetic",
        endpoint="bp",
        hex_id="16",
        baseline_fuel=baseline,
        replacement_map=replacement_map,
        original_nonfuel=original_nonfuel,
        unexpected_burnable_changes=unexpected,
    )
    assert summary.n_original_nonfuel_pixels == 2
    assert summary.n_replaced_pixels == 2
    assert summary.replacement_fuel_ids == "8;14"


def test_intervention_layers_detects_burnable_to_nonfuel_changes() -> None:
    baseline = np.array(
        [
            [1, 0, 2],
            [3, 0, np.nan],
        ],
        dtype=np.float32,
    )
    scenario = np.array(
        [
            [0, 0, 2],
            [3, 0, np.nan],
        ],
        dtype=np.float32,
    )

    original_nonfuel, replacement_map, unexpected = intervention_layers(baseline, scenario)
    assert original_nonfuel.tolist() == [[False, True, False], [False, True, False]]
    assert replacement_map[0, 0] == pytest.approx(0)
    assert np.isnan(replacement_map[0, 1])
    assert unexpected.tolist() == [[True, False, False], [False, False, False]]

    summary = summarize_intervention(
        scenario="synthetic_random",
        endpoint="bp",
        hex_id="16",
        baseline_fuel=baseline,
        replacement_map=replacement_map,
        original_nonfuel=original_nonfuel,
        unexpected_burnable_changes=unexpected,
    )
    assert summary.n_original_nonfuel_pixels == 2
    assert summary.n_replaced_pixels == 1
    assert summary.replacement_fuel_ids == "0"


def test_intervention_layers_on_prediction_grid_loads_evaluated_fuel_artifacts(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "experiment"
    baseline_prediction_dir = experiment_dir / "predictions" / "baseline" / "bp"
    scenario_prediction_dir = experiment_dir / "predictions" / "insert" / "bp"
    profile = {
        "driver": "GTiff",
        "height": 2,
        "width": 3,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_origin(0, 2, 1, 1),
        "nodata": -9999.0,
    }

    def write_raster(path: Path, values: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(values.astype(np.float32), 1)

    prediction_values = np.ones((2, 3), dtype=np.float32)
    write_raster(baseline_prediction_dir / "predicted_hexels" / "hexel_16_predicted.tif", prediction_values)
    write_raster(scenario_prediction_dir / "predicted_hexels" / "hexel_16_predicted.tif", prediction_values)
    write_raster(
        fuel_intervention_raster_path(scenario_prediction_dir, "16", "baseline"),
        np.array([[1, 101, 2], [3, 101, 4]], dtype=np.float32),
    )
    write_raster(
        fuel_intervention_raster_path(scenario_prediction_dir, "16", "scenario"),
        np.array([[101, 101, 2], [3, 101, 4]], dtype=np.float32),
    )
    pd.DataFrame(
        [
            {"scenario": "baseline", "endpoint": "bp", "prediction_dir": baseline_prediction_dir},
            {"scenario": "insert", "endpoint": "bp", "prediction_dir": scenario_prediction_dir},
        ]
    ).to_csv(experiment_dir / "scenario_prediction_index.csv", index=False)

    baseline, original_nonfuel, replacement_map, burnable_changes = intervention_layers_on_prediction_grid(
        experiment_dir=experiment_dir,
        scenario="insert",
        endpoint="bp",
        hex_id="16",
    )

    assert baseline.tolist() == [[1.0, 0.0, 2.0], [3.0, 0.0, 4.0]]
    assert original_nonfuel.tolist() == [[False, True, False], [False, True, False]]
    assert replacement_map[0, 0] == 0.0
    assert np.isnan(replacement_map[0, 1])
    assert burnable_changes.tolist() == [[True, False, False], [False, False, False]]


def test_replace_nonfuel_with_burnable_uses_fixed_replacement() -> None:
    fuel = np.array([[1, 101, 2], [100, 2, -32768]], dtype=np.int32)
    edited, edit_mask, report = replace_nonfuel_with_burnable(fuel, [100, 101], 2)
    assert edit_mask.tolist() == [[False, True, False], [True, False, False]]
    assert edited.tolist() == [[1, 2, 2], [2, 2, -32768]]
    assert report.edited_pixels == 2
    assert report.replacement_fuel_id == 2


def test_fuel_edits_preserve_nan_support_pixels() -> None:
    fuel = np.array([[1.0, 101.0, np.nan], [100.0, 2.0, np.nan]], dtype=np.float32)
    edited, edit_mask, report = replace_nonfuel_with_burnable(fuel, [100, 101], 2)
    assert edit_mask.tolist() == [[False, True, False], [True, False, False]]
    assert np.isnan(edited[0, 2])
    assert np.isnan(edited[1, 2])
    assert edited[0, 1] == pytest.approx(2.0)
    assert edited[1, 0] == pytest.approx(2.0)
    assert report.original_burnable_pixels == 2


def test_replace_burnable_with_nonfuel_only_edits_burnable_pixels() -> None:
    fuel = np.array(
        [
            [1, 101, 2],
            [1, 2, -32768],
        ],
        dtype=np.int32,
    )
    insertion = np.ones_like(fuel, dtype=bool)
    edited, edit_mask, report = replace_burnable_with_nonfuel(fuel, [101], insertion, replacement_nonfuel_id=101)
    assert edit_mask.tolist() == [[True, False, True], [True, True, False]]
    assert np.all(edited[edit_mask] == 101)
    assert edited[0, 1] == 101
    assert edited[1, 2] == -32768
    assert report.edited_pixels == 4


def test_replace_burnable_with_fixed_only_edits_source_fuel_pixels() -> None:
    fuel = np.array([[1, 2, 101], [2, 100, -32768]], dtype=np.int32)
    edited, edit_mask, report = replace_burnable_with_fixed(fuel, [100, 101], [2], 620)
    assert edit_mask.tolist() == [[False, True, False], [True, False, False]]
    assert edited.tolist() == [[1, 620, 101], [620, 100, -32768]]
    assert report.edited_pixels == 2
    assert report.replacement_fuel_id == 620


def test_replace_burnable_with_fixed_rejects_nonfuel_source_or_replacement() -> None:
    fuel = np.array([[1, 2, 101]], dtype=np.int32)
    with pytest.raises(ValueError, match="must not overlap"):
        replace_burnable_with_fixed(fuel, [100, 101], [101], 620)
    with pytest.raises(ValueError, match="is a non-fuel ID"):
        replace_burnable_with_fixed(fuel, [100, 101], [2], 101)


def test_apply_fuel_edit_routes_burnable_to_burnable_fixed() -> None:
    fuel = np.array([[1, 2, 101]], dtype=np.int32)
    result = apply_fuel_edit(
        fuel,
        [100, 101],
        mode="burnable_to_burnable_fixed",
        scenario_name="c2_to_mixedwood_fixed",
        params={"source_fuel_ids": [2], "replacement_fuel_id": 620},
    )
    assert result.edit_mask.tolist() == [[False, True, False]]
    assert result.fuel.tolist() == [[1, 620, 101]]
    assert result.report.replacement_fuel_id == 620


def test_random_component_insertion_is_seeded_and_reaches_area_target() -> None:
    fuel = np.array(
        [
            [1, 1, 101, 2, 2],
            [1, 1, 101, 2, 2],
            [3, 101, 4, 4, 4],
            [3, 101, 4, 4, 4],
        ],
        dtype=np.int32,
    )

    result_a = replace_random_burnable_components_with_nonfuel(
        fuel,
        [101],
        replacement_nonfuel_id=101,
        target_burnable_area_fraction=0.25,
        seed=42,
    )
    result_b = replace_random_burnable_components_with_nonfuel(
        fuel,
        [101],
        replacement_nonfuel_id=101,
        target_burnable_area_fraction=0.25,
        seed=42,
    )
    edited, edit_mask, report, components = result_a

    assert np.array_equal(edit_mask, result_b[1])
    assert np.all(edited[edit_mask] == 101)
    assert report.edited_pixels >= int(np.ceil(report.original_burnable_pixels * 0.25))
    assert set(components["original_fuel_id"]).issubset({1, 2, 3, 4})


def test_apply_fuel_edit_burnable_to_nonfuel_respects_edit_mask() -> None:
    fuel = np.array([[1, 2, 101], [2, 1, 1]], dtype=np.int32)
    edit_mask = np.array([[True, False, True], [False, True, False]])
    result = apply_fuel_edit(
        fuel,
        [101],
        mode="burnable_to_nonfuel",
        scenario_name="masked_barrier",
        params={"insertion_mask": np.ones_like(fuel, dtype=bool), "replacement_nonfuel_id": 101, "edit_mask": edit_mask},
    )
    assert result.edit_mask.tolist() == [[True, False, False], [False, True, False]]
    assert result.fuel.tolist() == [[101, 2, 101], [2, 101, 1]]


def test_apply_fuel_edit_random_components_respects_edit_mask() -> None:
    fuel = np.array([[1, 1, 101, 2, 2], [1, 1, 101, 2, 2], [3, 3, 101, 4, 4]], dtype=np.int32)
    edit_mask = np.zeros_like(fuel, dtype=bool)
    edit_mask[:, 3:] = True
    result = apply_fuel_edit(
        fuel,
        [101],
        mode="burnable_components_to_nonfuel_random",
        scenario_name="masked_random_barrier",
        params={"replacement_nonfuel_id": 101, "target_burnable_area_fraction": 1.0, "seed": 0, "edit_mask": edit_mask},
    )
    assert np.array_equal(result.edit_mask, edit_mask)
    assert np.array_equal(result.fuel[:, :3], fuel[:, :3])
    assert result.report.candidate_pixels == int(edit_mask.sum())


def test_zone_boundary_segments_traces_only_valid_interzone_borders() -> None:
    labels = np.ma.masked_array(
        np.array([[1, 1, 2], [1, 3, 2], [1, 3, 2]], dtype=np.int64),
        mask=np.zeros((3, 3), dtype=bool),
    )
    segments = zone_boundary_segments(labels)
    edges = {(tuple(np.round(a, 3)), tuple(np.round(b, 3))) for a, b in segments}
    expected = {
        ((0.5, 0.5), (0.5, 1.5)),
        ((0.5, 1.5), (0.5, 2.5)),
        ((1.5, -0.5), (1.5, 0.5)),
        ((1.5, 0.5), (1.5, 1.5)),
        ((1.5, 1.5), (1.5, 2.5)),
        ((0.5, 0.5), (1.5, 0.5)),
    }
    assert edges == expected


def test_zone_boundary_segments_skips_masked_neighbours() -> None:
    labels = np.ma.masked_array(
        np.array([[1, 2], [1, 2]], dtype=np.int64),
        mask=np.array([[False, True], [False, False]], dtype=bool),
    )
    segments = zone_boundary_segments(labels)
    assert segments.shape[0] == 1
    (start, end) = segments[0]
    assert tuple(np.round(start, 3)) == (0.5, 0.5)
    assert tuple(np.round(end, 3)) == (0.5, 1.5)

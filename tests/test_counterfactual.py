from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.datasets.postprocessing.counterfactual import (
    ScenarioConfig,
    encoded_components_from_from_bearing,
    load_counterfactual_config,
)
from src.datasets.postprocessing.counterfactual_compare import paired_delta_summary
from src.datasets.postprocessing.counterfactual_fi_map import distance_binned_delta_map
from src.datasets.postprocessing.counterfactual_fuel import (
    burnable_mask,
    modal_adjacent_burnable_fuel,
    modal_adjacent_burnable_fuel_across_grids,
    nonfuel_mask,
    replace_burnable_with_nonfuel,
    replace_nonfuel_components_with_adjacent_modal,
    replace_nonfuel_with_adjacent_modal,
    replace_nonfuel_with_burnable,
)
from src.datasets.postprocessing.counterfactual_fuel_intervention_map import (
    intervention_layers,
    summarize_intervention,
)
from src.datasets.postprocessing.counterfactual_hazard_map import (
    restrict_to_support,
    summarize_hazard_delta,
    symmetric_percentile_limit,
)
from src.datasets.postprocessing.counterfactual_local_zoom_panels import (
    select_neighborhood_windows,
)
from src.datasets.postprocessing.counterfactual_materialize import (
    _merge_materialized_index,
    build_scenario_endpoint_config,
    prepared_nonfuel_ids,
)
from src.datasets.postprocessing.counterfactual_viz import (
    abs_share_at,
    cumulative_abs_share,
    zone_boundary_segments,
)
from src.datasets.postprocessing.counterfactual_weather import (
    encode_raw_wind_features,
    raw_wind_features,
    recover_wind_encoding_stats,
    validate_wind_roundtrip,
)
from src.datasets.postprocessing.fuel_barrier_geometry import (
    SECTOR_ALL,
    SECTOR_CROSSWIND,
    cumulative_hazard_delta_share,
    hazard_decomposition_from_arrays,
    profile_from_arrays,
    relative_hazard_decomposition_summary,
)


def test_hex16_counterfactual_config_loads() -> None:
    cfg = load_counterfactual_config(Path("configs/counterfactual_hex16.yaml"))
    assert cfg.hex_ids == ["16"]
    assert cfg.focus_hex_id == "16"
    assert cfg.mask_scope == "actual"
    assert set(cfg.enabled_endpoints) == {"bp", "ros", "fi"}
    assert any(s.kind == "baseline" for s in cfg.scenarios)


def test_composite_scenario_resolves_fuel_and_weather_edits() -> None:
    scenario = ScenarioConfig.from_mapping(
        {
            "name": "remove_barriers_wind_opposite",
            "kind": "composite",
            "params": {
                "edits": [
                    {"kind": "fuel", "params": {"mode": "nonfuel_to_burnable_local_adjacent_modal"}},
                    {"kind": "wind_direction", "params": {"mode": "uniform_direction", "offset_deg": 180.0}},
                ]
            },
        }
    )
    assert scenario.fuel_edit() == {"mode": "nonfuel_to_burnable_local_adjacent_modal"}
    assert scenario.weather_edit() == ("wind_direction", {"mode": "uniform_direction", "offset_deg": 180.0})


def test_non_composite_scenario_resolves_only_matching_edit() -> None:
    scenario = ScenarioConfig.from_mapping({"name": "fwi_daily_low_to_high", "kind": "fwi", "params": {"mode": "daily_swap"}})
    assert scenario.edit_specs() == [("fwi", {"mode": "daily_swap"})]
    assert scenario.fuel_edit() is None
    assert scenario.weather_edit() == ("fwi", {"mode": "daily_swap"})


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"edits": []}, "non-empty"),
        ({"edits": [{"params": {}}]}, "without a 'kind'"),
        ({"edits": [{"kind": "bogus"}]}, "must be one of"),
        (
            {"edits": [{"kind": "fwi"}, {"kind": "wind_direction"}]},
            "more than one weather edit",
        ),
    ],
)
def test_composite_scenario_validation_rejects_bad_edits(params: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ScenarioConfig.from_mapping({"name": "bad", "kind": "composite", "params": params})


def test_from_bearing_components_point_to_source() -> None:
    x, y = encoded_components_from_from_bearing(10.0, 270.0)
    assert float(x) == pytest.approx(-10.0)
    assert float(y) == pytest.approx(0.0, abs=1e-12)


def test_wind_roundtrip_recovers_zscore_stats() -> None:
    raw_weather = pd.DataFrame(
        {
            "WindSpeed": [2.0, 6.0, 10.0, 14.0],
            "WindDirection": [270.0, 270.0, 90.0, 0.0],
        }
    )
    raw_features = raw_wind_features(raw_weather)
    processed = pd.DataFrame(
        {
            "WindSpeed": (raw_features["WindSpeed"] - 8.0) / 4.0,
            "wind_x": (raw_features["wind_x"] - 1.5) / 3.0,
            "wind_y": (raw_features["wind_y"] + 2.0) / 5.0,
        }
    )
    stats, report = validate_wind_roundtrip(raw_weather, processed)
    stats_by_name = stats.by_name()
    assert stats_by_name["WindSpeed"].mean == pytest.approx(8.0)
    assert stats_by_name["WindSpeed"].std == pytest.approx(4.0)
    assert stats_by_name["wind_x"].mean == pytest.approx(1.5)
    assert stats_by_name["wind_x"].std == pytest.approx(3.0)
    assert stats_by_name["wind_y"].mean == pytest.approx(-2.0)
    assert stats_by_name["wind_y"].std == pytest.approx(5.0)
    assert report["max_abs_error"].max() == pytest.approx(0.0, abs=1e-12)


def test_encode_raw_wind_features_uses_recovered_stats() -> None:
    raw_weather = pd.DataFrame(
        {
            "WindSpeed": [5.0, 7.0, 9.0],
            "WindDirection": [0.0, 90.0, 180.0],
        }
    )
    raw_features = raw_wind_features(raw_weather)
    processed = pd.DataFrame(
        {
            "WindSpeed": (raw_features["WindSpeed"] - 7.0) / 2.0,
            "wind_x": (raw_features["wind_x"] - 0.0) / 4.0,
            "wind_y": (raw_features["wind_y"] - 1.0) / 6.0,
        }
    )
    stats = recover_wind_encoding_stats(raw_features, processed)
    encoded = encode_raw_wind_features(raw_features, stats)
    pd.testing.assert_frame_equal(encoded, processed)


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


def test_select_neighborhood_windows_uses_positive_delta_and_barrier_density() -> None:
    barrier = np.zeros((30, 30), dtype=bool)
    barrier[2:8, 2:8] = True
    barrier[16:22, 16:22] = True
    valid = np.ones_like(barrier, dtype=bool)
    delta_hazard = np.ones((30, 30), dtype=float)
    delta_fi = np.ones((30, 30), dtype=float)
    delta_hazard[15:25, 15:25] = 5.0
    delta_fi[15:25, 15:25] = 10.0

    windows = select_neighborhood_windows(
        barrier_mask=barrier,
        valid_mask=valid,
        delta_hazard=delta_hazard,
        delta_fi=delta_fi,
        n_windows=1,
        crop_size=10,
        stride=5,
        min_barrier_pixels=10,
        min_barrier_density=0.05,
        max_barrier_density=0.8,
        exclude_border_pixels=0,
    )
    assert len(windows) == 1
    window = windows[0]
    assert window.row_min < 22 and window.row_max > 16
    assert window.col_min < 22 and window.col_max > 16
    assert window.delta_hazard_mean > 1.0


def test_modal_adjacent_burnable_across_grids_picks_consistent_replacement() -> None:
    grids = [
        np.array([[1, 101, 2], [1, 101, 2]], dtype=np.int32),
        np.array([[2, 101, 2], [3, 3, 2]], dtype=np.int32),
    ]
    replacement, candidate_pixels, note = modal_adjacent_burnable_fuel_across_grids(grids, [101])
    assert replacement == 2
    assert candidate_pixels > 0
    assert "across grids" in note


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


def test_build_scenario_endpoint_config_rewrites_data_and_save_paths(tmp_path: Path) -> None:
    endpoint_config = {
        "save_dir": "experiments/original",
        "logger": {"enabled": True},
        "data": {
            "root_dir": "data/original",
            "raw_data_dir": "raw/original",
            "train_split": "train_indices.csv",
            "val_split": "val_indices.csv",
            "test_split": "test_indices.csv",
            "input_sources": [
                {
                    "name": "spatialized_weather",
                    "params": {
                        "csv_name": "weather_table_processed.csv",
                        "feature_names_list": ["Temperature"],
                        "missing_value_strategy": "global_mean",
                    },
                }
            ],
        },
    }
    scenario_config = build_scenario_endpoint_config(
        endpoint_config,
        data_root=tmp_path / "data_root",
        prediction_dir=tmp_path / "predictions",
        raw_data_dir=tmp_path / "raw",
    )
    assert scenario_config["save_dir"] == str(tmp_path / "predictions")
    assert scenario_config["logger"]["enabled"] is False
    assert scenario_config["data"]["root_dir"] == str(tmp_path / "data_root")
    assert scenario_config["data"]["raw_data_dir"] == str(tmp_path / "raw")
    assert "imputation_stats_path" not in scenario_config["data"]["input_sources"][0]["params"]


def test_prepared_nonfuel_ids_maps_raw_fbp_to_grouped_patch_ids() -> None:
    assert prepared_nonfuel_ids([100, 101, 102, 105, 106, 110]) == [0]


def test_merge_materialized_index_replaces_only_updated_rows() -> None:
    existing = pd.DataFrame(
        [
            {"scenario": "baseline", "endpoint": "bp", "prediction_dir": "old/base/bp"},
            {"scenario": "baseline", "endpoint": "fi", "prediction_dir": "old/base/fi"},
            {"scenario": "fuel", "endpoint": "bp", "prediction_dir": "old/fuel/bp"},
        ]
    )
    updates = pd.DataFrame(
        [
            {"scenario": "fuel", "endpoint": "bp", "prediction_dir": "new/fuel/bp"},
            {"scenario": "fuel", "endpoint": "fi", "prediction_dir": "new/fuel/fi"},
        ]
    )

    merged = _merge_materialized_index(existing, updates)
    by_key = {(str(row.scenario), str(row.endpoint)): str(row.prediction_dir) for row in merged.itertuples(index=False)}
    assert by_key == {
        ("baseline", "bp"): "old/base/bp",
        ("baseline", "fi"): "old/base/fi",
        ("fuel", "bp"): "new/fuel/bp",
        ("fuel", "fi"): "new/fuel/fi",
    }


def test_paired_delta_summary_uses_finite_intersection() -> None:
    baseline = np.ma.masked_invalid(np.array([[1.0, 2.0], [np.nan, 4.0]]))
    scenario = np.ma.masked_invalid(np.array([[2.0, 1.0], [3.0, np.nan]]))
    summary = paired_delta_summary(
        baseline,
        scenario,
        scenario_name="synthetic",
        endpoint="bp",
        hex_id="16",
    )
    assert summary.n_pixels == 2
    assert summary.delta_mean == pytest.approx(0.0)
    assert summary.delta_abs_mean == pytest.approx(1.0)
    assert summary.delta_max_abs == pytest.approx(1.0)
    assert summary.frac_delta_positive == pytest.approx(0.5)
    assert summary.frac_delta_negative == pytest.approx(0.5)


def test_barrier_relative_profile_bins_deltas_by_distance_and_sector() -> None:
    baseline = np.array([[1.0, 2.0], [3.0, np.nan]])
    scenario = np.array([[2.0, 2.5], [2.0, 5.0]])
    analysis = np.array([[True, True], [True, True]])
    dist_m = np.array([[50.0, 150.0], [150.0, 50.0]])
    sectors = np.array(
        [
            ["downwind_of_barrier", SECTOR_CROSSWIND],
            ["upwind_of_barrier", "downwind_of_barrier"],
        ],
        dtype=object,
    )

    profile = profile_from_arrays(
        baseline=baseline,
        scenario=scenario,
        analysis_mask=analysis,
        dist_m=dist_m,
        sector_labels=sectors,
        bin_edges_m=(100.0,),
        scenario_name="synthetic",
        endpoint="bp",
        hex_id="16",
        sector_source="test",
    )

    all_near = profile[profile["sector"].eq(SECTOR_ALL) & profile["dist_bin_idx"].eq(0)].iloc[0]
    assert all_near["n_pixels"] == 1
    assert all_near["delta_mean"] == pytest.approx(1.0)

    cross_far = profile[profile["sector"].eq(SECTOR_CROSSWIND) & profile["dist_bin_idx"].eq(1)].iloc[0]
    assert cross_far["n_pixels"] == 1
    assert cross_far["delta_mean"] == pytest.approx(0.5)


def test_hazard_decomposition_terms_sum_to_delta_hazard() -> None:
    baseline_bp = np.array([[0.1, 0.2]])
    baseline_fi = np.array([[10.0, 20.0]])
    scenario_bp = np.array([[0.2, 0.1]])
    scenario_fi = np.array([[12.0, 18.0]])
    decomposition = hazard_decomposition_from_arrays(
        baseline_bp=baseline_bp,
        baseline_fi=baseline_fi,
        scenario_bp=scenario_bp,
        scenario_fi=scenario_fi,
        analysis_mask=np.ones((1, 2), dtype=bool),
        dist_m=np.array([[50.0, 150.0]]),
        sector_labels=np.array([[SECTOR_CROSSWIND, SECTOR_CROSSWIND]], dtype=object),
        bin_edges_m=(100.0,),
        scenario_name="synthetic",
        hex_id="16",
        sector_source="test",
    )
    all_rows = decomposition[decomposition["sector"].eq(SECTOR_ALL)].sort_values("dist_bin_idx")
    for row in all_rows.itertuples(index=False):
        total_terms = row.bp_component_mean + row.fi_component_mean + row.interaction_component_mean
        assert row.delta_hazard_mean == pytest.approx(total_terms)


def test_relative_hazard_decomposition_uses_baseline_hazard_sums() -> None:
    baseline_bp = np.array([[0.1, 0.2]])
    baseline_fi = np.array([[10.0, 20.0]])
    scenario_bp = np.array([[0.2, 0.3]])
    scenario_fi = np.array([[12.0, 20.0]])
    decomposition = hazard_decomposition_from_arrays(
        baseline_bp=baseline_bp,
        baseline_fi=baseline_fi,
        scenario_bp=scenario_bp,
        scenario_fi=scenario_fi,
        analysis_mask=np.ones((1, 2), dtype=bool),
        dist_m=np.array([[50.0, 50.0]]),
        sector_labels=np.array([[SECTOR_CROSSWIND, SECTOR_CROSSWIND]], dtype=object),
        bin_edges_m=(100.0,),
        scenario_name="remove_barriers_adjacent_modal",
        hex_id="16",
        sector_source="test",
    )

    summary = relative_hazard_decomposition_summary(decomposition)

    row = summary[summary["sector"].eq(SECTOR_ALL)].iloc[0]
    assert row["baseline_hazard_sum"] == pytest.approx(5.0)
    assert row["relative_bp_component_percent"] == pytest.approx(60.0)
    assert row["relative_fi_component_percent"] == pytest.approx(4.0)
    assert row["relative_interaction_component_percent"] == pytest.approx(4.0)
    assert row["relative_delta_hazard_percent"] == pytest.approx(68.0)


def test_cumulative_hazard_delta_share_sums_distance_bins() -> None:
    baseline = np.zeros((1, 4), dtype=float)
    scenario = np.array([[1.0, 3.0, 6.0, 10.0]])
    profile = profile_from_arrays(
        baseline=baseline,
        scenario=scenario,
        analysis_mask=np.ones((1, 4), dtype=bool),
        dist_m=np.array([[50.0, 150.0, 300.0, 700.0]]),
        sector_labels=np.full((1, 4), SECTOR_CROSSWIND, dtype=object),
        bin_edges_m=(100.0, 250.0, 500.0),
        scenario_name="remove_barriers_adjacent_modal",
        endpoint="hazard_bp_x_fi",
        hex_id="16",
        sector_source="test",
    )

    summary = cumulative_hazard_delta_share(profile, thresholds_m=(250.0, 500.0))

    within_250 = summary[summary["threshold_m"].eq(250.0)].iloc[0]
    assert within_250["delta_hazard_sum_within"] == pytest.approx(4.0)
    assert within_250["percent_of_total_delta_hazard"] == pytest.approx(20.0)

    within_500 = summary[summary["threshold_m"].eq(500.0)].iloc[0]
    assert within_500["delta_hazard_sum_within"] == pytest.approx(10.0)
    assert within_500["percent_of_total_delta_hazard"] == pytest.approx(50.0)
    assert within_500["percent_of_positive_delta_hazard"] == pytest.approx(50.0)


def test_hazard_delta_map_summary_uses_paired_support() -> None:
    baseline = np.ma.masked_invalid(np.array([[1.0, 2.0], [np.nan, 4.0]]))
    scenario = np.ma.masked_invalid(np.array([[2.0, 1.0], [3.0, np.nan]]))
    delta = scenario - baseline
    summary = summarize_hazard_delta(
        scenario="synthetic",
        hex_id="16",
        baseline_hazard=baseline,
        scenario_hazard=scenario,
        delta=delta,
        delta_abs_plot_limit=1.0,
    )
    assert summary.n_pixels == 2
    assert summary.baseline_hazard_mean == pytest.approx(1.5)
    assert summary.scenario_hazard_mean == pytest.approx(1.5)
    assert summary.delta_mean == pytest.approx(0.0)
    assert summary.delta_abs_plot_limit == pytest.approx(1.0)
    assert summary.support_policy == "prediction"


def test_restrict_to_support_masks_outside_support() -> None:
    data = np.ma.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    support = np.array([[True, False], [False, True]])
    restricted = restrict_to_support(data, support)
    assert restricted.mask.tolist() == [[False, True], [True, False]]
    assert restricted.filled(np.nan)[0, 0] == pytest.approx(1.0)


def test_symmetric_percentile_limit_uses_absolute_pooled_deltas() -> None:
    deltas = [
        np.ma.asarray(np.array([-1.0, 2.0, np.nan])),
        np.ma.asarray(np.array([4.0, -8.0])),
    ]
    assert symmetric_percentile_limit(deltas, percentile=100.0) == pytest.approx(8.0)


def test_distance_binned_delta_map_assigns_bin_means() -> None:
    delta = np.ma.masked_invalid(np.array([[1.0, 3.0], [5.0, np.nan]]))
    dist_m = np.array([[50.0, 150.0], [150.0, 50.0]])
    support = np.ones((2, 2), dtype=bool)
    binned, summary = distance_binned_delta_map(delta, dist_m, support, bin_edges_m=(100.0,))
    filled = binned.filled(np.nan)
    assert filled[0, 0] == pytest.approx(1.0)
    assert filled[0, 1] == pytest.approx(4.0)
    assert filled[1, 0] == pytest.approx(4.0)
    assert np.isnan(filled[1, 1])
    assert summary["n_pixels"].tolist() == [1, 2]
    assert summary["delta_fi_mean"].tolist() == pytest.approx([1.0, 4.0])


def test_cumulative_abs_share_uniform_delta_tracks_diagonal() -> None:
    delta = np.ma.masked_array(np.array([2.0, -2.0, 2.0, -2.0]), mask=False)
    pixel_fraction, cumulative = cumulative_abs_share(delta)
    assert pixel_fraction.tolist() == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert cumulative.tolist() == pytest.approx([0.25, 0.5, 0.75, 1.0])


def test_cumulative_abs_share_concentrated_delta_bows_to_top_left() -> None:
    delta = np.ma.masked_array(np.array([100.0, 1.0, 1.0, 1.0, 1.0]), mask=False)
    pixel_fraction, cumulative = cumulative_abs_share(delta)
    assert abs_share_at(pixel_fraction, cumulative, 0.2) == pytest.approx(100.0 / 104.0)


def test_cumulative_abs_share_ignores_masked_and_handles_empty() -> None:
    delta = np.ma.masked_array(np.array([5.0, np.nan, -3.0]), mask=[False, True, False])
    pixel_fraction, cumulative = cumulative_abs_share(delta)
    assert cumulative[-1] == pytest.approx(1.0)
    assert abs_share_at(pixel_fraction, cumulative, 0.5) == pytest.approx(5.0 / 8.0)
    empty_fraction, empty_cumulative = cumulative_abs_share(np.ma.masked_array(np.array([np.nan]), mask=[True]))
    assert abs_share_at(empty_fraction, empty_cumulative, 0.1) == pytest.approx(0.0)


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

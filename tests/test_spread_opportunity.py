from pathlib import Path

import pandas as pd
import pytest

from data_preparation.process_spread_opportunity import (
    NORMALIZED_QUANTILE_COLUMNS,
    RAW_QUANTILE_COLUMNS,
    add_shared_minmax_columns,
    build_hex_spread_opportunity_from_tables,
    discrete_quantile,
    mix_pmfs,
    multiply_pmfs,
    normalize_pmf,
    read_scenario_distributions,
)


def test_pmf_helpers_normalize_mix_and_multiply():
    days = normalize_pmf([(1, 35), (2, 29), (3, 20), (4, 16), (5, 0)], context="days")
    hours = {4.0: 1.0}

    total_hours = multiply_pmfs(days, hours, context="total")
    mixture = mix_pmfs([(1.0, total_hours), (1.0, {6.0: 1.0})], context="mixture")

    assert sum(total_hours.values()) == pytest.approx(1.0)
    assert total_hours == pytest.approx({4.0: 0.35, 8.0: 0.29, 12.0: 0.20, 16.0: 0.16})
    assert discrete_quantile(mixture, 0.5) == 6.0


def test_hex01_fru21_fixture_matches_expected_q3_and_mean():
    spread_pmfs = {
        ("fru21", "s1"): normalize_pmf(zip([1, 2, 3, 4], [35, 29, 20, 16], strict=True), context="s1"),
        ("fru21", "s2"): normalize_pmf(
            zip([1, 2, 3, 4, 5, 6, 7], [65, 17.5, 7.5, 3.5, 3, 2, 1.5], strict=True),
            context="s2",
        ),
    }
    frame = build_hex_spread_opportunity_from_tables(
        hex_id="01",
        spread_day_pmfs=spread_pmfs,
        daily_hour_pmfs={"s1": {4.0: 1.0}, "s2": {6.0: 1.0}},
        likelihoods={("fru21", "s1"): 1.0, ("fru21", "s2"): 1.5},
        name_to_id={"fru21": 21},
        id_to_name={21: "fru21"},
        area_fractions={21: 1.0},
    )
    row = frame.iloc[0]

    assert row["TOTAL_BURN_HOURS_MEAN"] == pytest.approx(9.736)
    assert row["TOTAL_BURN_HOURS_Q10"] == 4.0
    assert row["TOTAL_BURN_HOURS_Q50"] == 6.0
    assert row["TOTAL_BURN_HOURS_Q90"] == 18.0
    assert row["SCENARIO_FALLBACK"] == 0


def test_missing_direct_zone_uses_area_weighted_fallback(caplog):
    frame = build_hex_spread_opportunity_from_tables(
        hex_id="01",
        spread_day_pmfs={("fru21", "s1"): {2.0: 1.0}},
        daily_hour_pmfs={"s1": {4.0: 1.0}},
        likelihoods={("fru21", "s1"): 2.0, ("fru25", "s1"): 1.0},
        name_to_id={"fru21": 21, "fru25": 25},
        id_to_name={21: "fru21", 25: "fru25"},
        area_fractions={21: 0.25, 25: 0.75},
    ).set_index("GRIDCODE")

    assert frame.loc[21, "SCENARIO_FALLBACK"] == 0
    assert frame.loc[25, "SCENARIO_FALLBACK"] == 1
    assert frame.loc[25, "TOTAL_BURN_HOURS_Q50"] == 8.0
    assert frame.loc[25, "SCENARIO_DROPPED_LIKELIHOOD"] == 1.0
    assert "dropping positive scenario likelihood" in caplog.text


def test_unmapped_numeric_raster_zone_can_receive_fallback():
    frame = build_hex_spread_opportunity_from_tables(
        hex_id="46",
        spread_day_pmfs={("fru05", "s1"): {2.0: 1.0}},
        daily_hour_pmfs={"s1": {4.0: 1.0}},
        likelihoods={("fru05", "s1"): 1.0},
        name_to_id={"fru05": 5, "unmapped_gridcode_1": 1},
        id_to_name={5: "fru05", 1: "unmapped_gridcode_1"},
        area_fractions={5: 0.9, 1: 0.1},
    ).set_index("GRIDCODE")

    assert frame.loc[1, "SCENARIO_FALLBACK"] == 1
    assert frame.loc[1, "TOTAL_BURN_HOURS_Q50"] == 8.0


def test_hex28_malformed_scenario_header_is_accepted(tmp_path: Path, caplog):
    path = tmp_path / "scenario.csv"
    path.write_text("i wantetatName,Value,RelativeFrequency\nspread,1,40\nspread,2,60\n")

    frame = read_scenario_distributions(path)

    assert list(frame.columns) == ["Name", "Value", "RelativeFrequency"]
    assert frame["Name"].tolist() == ["spread", "spread"]
    assert "malformed first column" in caplog.text


def test_shared_minmax_is_fit_on_training_hexes_without_clipping(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "hex_id": [1, 2],
            "TOTAL_BURN_HOURS_MEAN": [15.0, 30.0],
            "TOTAL_BURN_HOURS_Q10": [10.0, 20.0],
            "TOTAL_BURN_HOURS_Q50": [15.0, 30.0],
            "TOTAL_BURN_HOURS_Q90": [20.0, 40.0],
        }
    )
    params_path = tmp_path / "params.json"

    normalized, fitted_range = add_shared_minmax_columns(
        frame,
        train_hex_ids={1},
        norm_params_path=params_path,
    )

    assert fitted_range == (10.0, 20.0)
    assert normalized.loc[0, NORMALIZED_QUANTILE_COLUMNS[0.1]] == 0.0
    assert normalized.loc[0, NORMALIZED_QUANTILE_COLUMNS[0.9]] == 1.0
    assert normalized.loc[1, NORMALIZED_QUANTILE_COLUMNS[0.9]] == 3.0
    assert params_path.exists()
    assert [RAW_QUANTILE_COLUMNS[q] for q in (0.1, 0.5, 0.9)] == [
        "TOTAL_BURN_HOURS_Q10",
        "TOTAL_BURN_HOURS_Q50",
        "TOTAL_BURN_HOURS_Q90",
    ]

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.datasets.postprocessing.counterfactual import counterfactual_weather as cw
from src.datasets.postprocessing.counterfactual.counterfactual_base import ScenarioConfig
from src.datasets.postprocessing.counterfactual.weather_counterfactual_transform import (
    materialize_weather_scenario,
    weather_intervention_csv_path,
)


def _weather_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.DataFrame(
        {
            "__hex_id": ["16", "16", "17", "17", "01", "01"],
            "WeatherZone": [4, 9, 30, 31, 4, 4],
            "FireWeatherIndex": [10.0, 20.0, 30.0, 50.0, 5.0, 7.0],
            "WindSpeed": [5.0, 25.0, 10.0, 40.0, 8.0, 12.0],
        }
    )
    processed = pd.DataFrame(
        {
            "Order": [1, 2, 3, 4, 5, 6],
            "Season": [1, 2, 1, 2, 1, 2],
            "hex_id": [16, 16, 17, 17, 1, 1],
            "WeatherZone": raw["WeatherZone"],
            "Temperature": [-1.0, 0.0, 1.0, 3.0, -2.0, -4.0],
            "WindDirection": [180.0, 200.0, 220.0, 240.0, 90.0, 270.0],
            "FireWeatherIndex": [-1.0, 0.0, 2.0, 4.0, -2.0, -4.0],
            "wind_x": [-0.5, -0.25, 0.5, 1.5, 0.0, -1.0],
            "wind_y": [0.0, 0.25, 1.0, 2.0, -0.5, 0.5],
        }
    )
    return raw, processed


def _write_norm_params(path: Path) -> None:
    """Write minimal weather normalization params: wind_x/wind_y z-score only."""
    payload = {
        "min_max": {"cols": [], "min": [], "max": []},
        "z_score": {"cols": ["wind_x", "wind_y"], "mean": [1.0, -1.0], "std": [2.0, 0.5]},
    }
    path.write_text(json.dumps(payload))


def test_apply_external_mean_zone_transplant_builds_exact_recipient_hex_zone_lut() -> None:
    raw, processed = _weather_frames()

    edited, reports = cw.apply_external_mean_zone_transplant(
        raw,
        processed,
        recipient_hex_ids=["16"],
        donor_hex_ids=["17"],
        scenario_name="bc_mean_weather_transplant",
    )

    assert len(edited) == processed[["hex_id", "WeatherZone"]].drop_duplicates().shape[0]
    donor_mean = processed.loc[[2, 3], ["Temperature", "FireWeatherIndex", "wind_x", "wind_y"]].mean()
    for zone in (4, 9):
        row = edited.loc[(edited["hex_id"] == 16) & (edited["WeatherZone"] == zone)].iloc[0]
        assert row[donor_mean.index].to_numpy(dtype=np.float64) == pytest.approx(donor_mean.to_numpy(dtype=np.float64))

    untouched = edited.loc[(edited["hex_id"] == 1) & (edited["WeatherZone"] == 4)].iloc[0]
    assert untouched["FireWeatherIndex"] == pytest.approx(processed.loc[[4, 5], "FireWeatherIndex"].mean())
    assert {"hex_id", "WeatherZone"} <= set(edited.columns)
    assert not (set(cw.NON_AVERAGE_COLUMNS) - {"hex_id", "WeatherZone"}).intersection(edited.columns)

    assert len(reports) == 1
    report = reports[0]
    assert report.mode == "external_mean_zone_transplant"
    assert report.recipient_hex_id == "16"
    assert report.n_recipient_rows == 2
    assert report.n_recipient_zones == 2
    assert report.donor_hex_ids == "17"
    assert report.n_donor_rows == 2
    assert report.donor_fwi_mean == pytest.approx(40.0)
    assert report.baseline_fwi_mean == pytest.approx(15.0)
    assert report.scenario_fwi_mean == pytest.approx(40.0)


@pytest.mark.parametrize(
    ("recipient_hex_ids", "donor_hex_ids", "message"),
    [
        (["99"], ["17"], "No weather rows found for recipient"),
        (["16"], ["99"], "No weather rows found for donor"),
        ([], ["17"], "recipient_hex_ids must be non-empty"),
        (["16"], [], "donor_hex_ids must be non-empty"),
    ],
)
def test_apply_external_mean_zone_transplant_validates_hex_ids(
    recipient_hex_ids: list[str],
    donor_hex_ids: list[str],
    message: str,
) -> None:
    raw, processed = _weather_frames()

    with pytest.raises(ValueError, match=message):
        cw.apply_external_mean_zone_transplant(
            raw,
            processed,
            recipient_hex_ids=recipient_hex_ids,
            donor_hex_ids=donor_hex_ids,
            scenario_name="scenario",
        )


def test_apply_external_mean_zone_transplant_rejects_row_count_mismatch() -> None:
    raw, processed = _weather_frames()
    with pytest.raises(ValueError, match="row count mismatch"):
        cw.apply_external_mean_zone_transplant(
            raw.iloc[:-1],
            processed,
            recipient_hex_ids=["16"],
            donor_hex_ids=["17"],
            scenario_name="scenario",
        )


@pytest.mark.parametrize(
    ("column", "values", "message"),
    [
        ("hex_id", [16, 16, 17, 17, 2, 2], "hex_id mismatch"),
        ("WeatherZone", [4, 9, 30, 31, 5, 5], "WeatherZone mismatch"),
    ],
)
def test_apply_external_mean_zone_transplant_rejects_row_misalignment(
    column: str,
    values: list[int],
    message: str,
) -> None:
    raw, processed = _weather_frames()
    processed[column] = values

    with pytest.raises(ValueError, match=message):
        cw.apply_external_mean_zone_transplant(
            raw,
            processed,
            recipient_hex_ids=["16"],
            donor_hex_ids=["17"],
            scenario_name="scenario",
        )


def test_apply_external_mean_zone_transplant_requires_processed_hex_id() -> None:
    raw, processed = _weather_frames()

    with pytest.raises(ValueError, match="missing column 'hex_id'"):
        cw.apply_external_mean_zone_transplant(
            raw,
            processed.drop(columns="hex_id"),
            recipient_hex_ids=["16"],
            donor_hex_ids=["17"],
            scenario_name="scenario",
        )


def test_apply_weather_edit_dispatches_mean_mode_and_rejects_invalid_configuration() -> None:
    raw, processed = _weather_frames()
    edited, _ = cw.apply_weather_edit(
        raw,
        processed,
        mode="external_mean_zone_transplant",
        scenario_name="scenario",
        recipient_hex_ids=["16"],
        params={"donor_hex_ids": ["17"]},
    )
    assert set(map(tuple, edited[["hex_id", "WeatherZone"]].to_numpy())) == set(
        map(tuple, processed[["hex_id", "WeatherZone"]].drop_duplicates().to_numpy())
    )

    with pytest.raises(ValueError, match="Unknown weather edit mode"):
        cw.apply_weather_edit(
            raw,
            processed,
            mode="not_a_mode",
            scenario_name="scenario",
            recipient_hex_ids=["16"],
            params={"donor_hex_ids": ["17"]},
        )
    with pytest.raises(ValueError, match="donor_hex_ids"):
        cw.apply_weather_edit(
            raw,
            processed,
            mode="external_mean_zone_transplant",
            scenario_name="scenario",
            recipient_hex_ids=["16"],
            params={},
        )


def _zone_dependent_weather_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Single self-donor hex ("16") with two zones (4, 9) that have distinctly
    different wind climatologies, to exercise per-zone (not hex-wide) donor pools.
    Each zone alternates Season 1/2 (lower/higher WindSpeed) to also exercise the
    per-zone `season` filter.
    """
    raw = pd.DataFrame(
        {
            "__hex_id": ["16", "16", "16", "16"],
            "WeatherZone": [4, 4, 9, 9],
            "FireWeatherIndex": [1.0, 2.0, 3.0, 4.0],
            "WindSpeed": [5.0, 15.0, 8.0, 40.0],
        }
    )
    processed = pd.DataFrame(
        {
            "Order": [1, 2, 3, 4],
            "Season": [1, 2, 1, 2],
            "hex_id": [16, 16, 16, 16],
            "WeatherZone": raw["WeatherZone"],
            "Temperature": [10.0, 20.0, 30.0, 40.0],
            "WindDirection": [0.0, 0.0, 0.0, 0.0],
            "FireWeatherIndex": [1.0, 2.0, 3.0, 4.0],
            "wind_x": [0.1, 0.2, 0.3, 0.4],
            "wind_y": [-0.1, -0.2, -0.3, -0.4],
        }
    )
    return raw, processed


def test_apply_windy_mean_zone_dependent_transplant_computes_per_zone_donor_means() -> None:
    raw, processed = _zone_dependent_weather_frames()

    # percentile=100 keeps only each zone's own windiest row: zone 4's WindSpeed=15
    # row (index 1) and zone 9's WindSpeed=40 row (index 3), independently.
    edited, reports = cw.apply_windy_mean_zone_dependent_transplant(
        raw,
        processed,
        recipient_hex_ids=["16"],
        donor_hex_ids=["16"],
        wind_speed_percentile=100.0,
        scenario_name="windy_zone_dependent",
    )

    zone4_row = edited.loc[(edited["hex_id"] == 16) & (edited["WeatherZone"] == 4)].iloc[0]
    zone9_row = edited.loc[(edited["hex_id"] == 16) & (edited["WeatherZone"] == 9)].iloc[0]
    expected_zone4 = processed.loc[1, ["Temperature", "FireWeatherIndex", "wind_x", "wind_y"]]
    expected_zone9 = processed.loc[3, ["Temperature", "FireWeatherIndex", "wind_x", "wind_y"]]
    assert zone4_row[expected_zone4.index].to_numpy(dtype=np.float64) == pytest.approx(expected_zone4.to_numpy(dtype=np.float64))
    assert zone9_row[expected_zone9.index].to_numpy(dtype=np.float64) == pytest.approx(expected_zone9.to_numpy(dtype=np.float64))
    # The two zones must get genuinely different donor means, unlike the whole-hex
    # mode which broadcasts one identical vector everywhere.
    assert zone4_row["FireWeatherIndex"] != pytest.approx(zone9_row["FireWeatherIndex"])

    assert len(reports) == 2
    reports_by_zone = {report.weather_zone: report for report in reports}
    assert set(reports_by_zone) == {4, 9}
    assert reports_by_zone[4].n_donor_rows == 1
    assert reports_by_zone[4].wind_speed_threshold_kmh == pytest.approx(15.0)
    assert reports_by_zone[9].n_donor_rows == 1
    assert reports_by_zone[9].wind_speed_threshold_kmh == pytest.approx(40.0)
    for report in reports:
        assert report.mode == "windy_mean_zone_dependent_transplant"
        assert report.wind_speed_percentile == pytest.approx(100.0)
        assert report.n_recipient_zones == 1


def test_apply_windy_mean_zone_dependent_transplant_rejects_out_of_range_percentile() -> None:
    raw, processed = _zone_dependent_weather_frames()

    with pytest.raises(ValueError, match="wind_speed_percentile must be within"):
        cw.apply_windy_mean_zone_dependent_transplant(
            raw,
            processed,
            recipient_hex_ids=["16"],
            donor_hex_ids=["16"],
            wind_speed_percentile=150.0,
            scenario_name="scenario",
        )


def test_apply_windy_mean_zone_dependent_transplant_requires_wind_speed_column() -> None:
    raw, processed = _zone_dependent_weather_frames()

    with pytest.raises(ValueError, match="missing 'WindSpeed'"):
        cw.apply_windy_mean_zone_dependent_transplant(
            raw.drop(columns="WindSpeed"),
            processed,
            recipient_hex_ids=["16"],
            donor_hex_ids=["16"],
            wind_speed_percentile=90.0,
            scenario_name="scenario",
        )


def test_apply_windy_mean_zone_dependent_transplant_filters_by_season_per_zone() -> None:
    raw, processed = _zone_dependent_weather_frames()

    # season=1 keeps only each zone's Season==1 row: zone 4's WindSpeed=5 row
    # (index 0) and zone 9's WindSpeed=8 row (index 2), independently.
    _, reports = cw.apply_windy_mean_zone_dependent_transplant(
        raw,
        processed,
        recipient_hex_ids=["16"],
        donor_hex_ids=["16"],
        wind_speed_percentile=0.0,
        season=1,
        scenario_name="windy_zone_dependent_s1",
    )
    reports_by_zone = {report.weather_zone: report for report in reports}
    assert reports_by_zone[4].n_donor_rows == 1
    assert reports_by_zone[4].wind_speed_threshold_kmh == pytest.approx(5.0)
    assert reports_by_zone[9].n_donor_rows == 1
    assert reports_by_zone[9].wind_speed_threshold_kmh == pytest.approx(8.0)
    for report in reports:
        assert report.season == 1


def test_apply_windy_mean_zone_dependent_transplant_raises_when_zone_has_no_matching_season() -> None:
    raw, processed = _zone_dependent_weather_frames()

    with pytest.raises(ValueError, match="No weather rows with Season=99"):
        cw.apply_windy_mean_zone_dependent_transplant(
            raw,
            processed,
            recipient_hex_ids=["16"],
            donor_hex_ids=["16"],
            wind_speed_percentile=0.0,
            season=99,
            scenario_name="scenario",
        )


def test_apply_wind_direction_zone_dependent_transplant_preserves_per_zone_magnitude(tmp_path: Path) -> None:
    raw, processed = _zone_dependent_weather_frames()
    norm_params_path = tmp_path / "weather_norm_params.json"
    _write_norm_params(norm_params_path)

    # percentile=0 keeps every donor row per zone; direction=90 (east) forces
    # raw wind_x=WindSpeed, wind_y=0 for every row before averaging within each zone.
    edited, reports = cw.apply_wind_direction_zone_dependent_transplant(
        raw,
        processed,
        recipient_hex_ids=["16"],
        donor_hex_ids=["16"],
        direction_degrees=90.0,
        wind_speed_percentile=0.0,
        norm_params_path=norm_params_path,
        scenario_name="wind_dir_090_zone_dependent",
    )

    zone4_row = edited.loc[(edited["hex_id"] == 16) & (edited["WeatherZone"] == 4)].iloc[0]
    zone9_row = edited.loc[(edited["hex_id"] == 16) & (edited["WeatherZone"] == 9)].iloc[0]
    # zone 4 mean WindSpeed=(5+15)/2=10 -> raw wind_x=10 -> normalized (10-1)/2=4.5
    assert zone4_row["wind_x"] == pytest.approx(4.5)
    assert zone4_row["wind_y"] == pytest.approx(2.0)
    assert zone4_row["Temperature"] == pytest.approx(15.0)
    # zone 9 mean WindSpeed=(8+40)/2=24 -> raw wind_x=24 -> normalized (24-1)/2=11.5
    assert zone9_row["wind_x"] == pytest.approx(11.5)
    assert zone9_row["Temperature"] == pytest.approx(35.0)
    # Same forced direction, but different per-zone wind magnitude survives.
    assert zone4_row["wind_x"] != pytest.approx(zone9_row["wind_x"])

    assert len(reports) == 2
    reports_by_zone = {report.weather_zone: report for report in reports}
    assert reports_by_zone[4].n_donor_rows == 2
    assert reports_by_zone[9].n_donor_rows == 2
    for report in reports:
        assert report.mode == "wind_direction_zone_dependent_transplant"
        assert report.direction_degrees == pytest.approx(90.0)


def test_apply_weather_edit_dispatches_zone_dependent_modes() -> None:
    raw, processed = _zone_dependent_weather_frames()

    _, windy_reports = cw.apply_weather_edit(
        raw,
        processed,
        mode="windy_mean_zone_dependent_transplant",
        scenario_name="scenario",
        recipient_hex_ids=["16"],
        params={"donor_hex_ids": ["16"], "wind_speed_percentile": 100.0},
    )
    assert {report.mode for report in windy_reports} == {"windy_mean_zone_dependent_transplant"}
    assert {report.weather_zone for report in windy_reports} == {4, 9}


def test_apply_weather_edit_dispatches_wind_direction_zone_dependent_mode(tmp_path: Path) -> None:
    raw, processed = _zone_dependent_weather_frames()
    norm_params_path = tmp_path / "weather_norm_params.json"
    _write_norm_params(norm_params_path)

    _, reports = cw.apply_weather_edit(
        raw,
        processed,
        mode="wind_direction_zone_dependent_transplant",
        scenario_name="scenario",
        recipient_hex_ids=["16"],
        params={"donor_hex_ids": ["16"], "direction_degrees": 90.0, "wind_speed_percentile": 0.0},
        norm_params_path=norm_params_path,
    )
    assert {report.mode for report in reports} == {"wind_direction_zone_dependent_transplant"}
    assert {report.weather_zone for report in reports} == {4, 9}


def test_weather_scenario_kind_and_weather_edit_accessor() -> None:
    scenario = ScenarioConfig(
        name="bc_mean_weather_transplant",
        kind="weather",
        description="",
        params={"mode": "external_mean_zone_transplant", "donor_hex_ids": ["17"]},
    )
    assert scenario.weather_edit() == {"mode": "external_mean_zone_transplant", "donor_hex_ids": ["17"]}
    assert scenario.fuel_edit() is None

    baseline = ScenarioConfig(name="baseline", kind="baseline", description="", params={})
    assert baseline.weather_edit() is None


def test_materialize_mean_weather_scenario_writes_compact_hex_zone_lut(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, processed = _weather_frames()
    processed_csv = tmp_path / "weather_table_processed.csv"
    processed.to_csv(processed_csv, index=False)
    monkeypatch.setattr(
        "src.datasets.postprocessing.counterfactual.weather_counterfactual_transform.load_all_raw_weather_with_hex_ids",
        lambda _raw_data_dir: raw,
    )
    scenario = ScenarioConfig(
        name="bc_mean_weather_transplant",
        kind="weather",
        description="",
        params={"mode": "external_mean_zone_transplant", "donor_hex_ids": ["17"]},
    )
    prediction_dir = tmp_path / "predictions" / scenario.name / "bp"

    result = materialize_weather_scenario(
        scenario=scenario,
        raw_data_dir=tmp_path / "raw",
        processed_weather_csv=processed_csv,
        recipient_hex_ids=["16"],
        prediction_dir=prediction_dir,
    )

    assert result.edited_csv_path == weather_intervention_csv_path(prediction_dir)
    edited = pd.read_csv(result.edited_csv_path)
    assert len(edited) == processed[["hex_id", "WeatherZone"]].drop_duplicates().shape[0]
    assert edited.loc[(edited["hex_id"] == 16) & edited["WeatherZone"].isin([4, 9]), "FireWeatherIndex"].to_numpy() == pytest.approx(
        [3.0, 3.0]
    )
    assert edited.loc[(edited["hex_id"] == 1) & (edited["WeatherZone"] == 4), "FireWeatherIndex"].item() == pytest.approx(-3.0)
    assert result.summary[["scenario_name", "donor_hex_ids"]].to_dict("records") == [
        {"scenario_name": "bc_mean_weather_transplant", "donor_hex_ids": "17"}
    ]


def test_materialize_weather_scenario_requires_explicit_mode(tmp_path: Path) -> None:
    processed_csv = tmp_path / "weather_table_processed.csv"
    pd.DataFrame({"hex_id": [1], "WeatherZone": [1], "FireWeatherIndex": [0.0]}).to_csv(processed_csv, index=False)
    scenario = ScenarioConfig(
        name="missing_mode",
        kind="weather",
        description="",
        params={"donor_hex_ids": ["17"]},
    )

    with pytest.raises(ValueError, match="explicit non-empty mode"):
        materialize_weather_scenario(
            scenario=scenario,
            raw_data_dir=tmp_path / "raw",
            processed_weather_csv=processed_csv,
            recipient_hex_ids=["16"],
            prediction_dir=tmp_path / "predictions",
        )


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_apply_windy_mean_zone_dependent_transplant_rejects_non_finite_donor_wind_speeds(bad_value: float) -> None:
    """A NaN/inf WindSpeed makes np.percentile return NaN, which would silently drop every donor row."""
    raw, processed = _zone_dependent_weather_frames()
    raw.loc[1, "WindSpeed"] = bad_value

    with pytest.raises(ValueError, match=r"WindSpeed has 1 missing/non-finite value\(s\) in the donor pool"):
        cw.apply_windy_mean_zone_dependent_transplant(
            raw,
            processed,
            recipient_hex_ids=["16"],
            donor_hex_ids=["16"],
            wind_speed_percentile=90.0,
            scenario_name="scenario",
        )


@pytest.mark.parametrize("bad_std", [0.0, -1.0, float("nan")])
def test_load_wind_component_norm_params_rejects_unusable_std(tmp_path: Path, bad_std: float) -> None:
    """A zero/negative/non-finite std must fail loudly instead of being coerced to 1.0."""
    path = tmp_path / "weather_norm_params.json"
    payload = {
        "min_max": {"cols": [], "min": [], "max": []},
        "z_score": {"cols": ["wind_x", "wind_y"], "mean": [1.0, -1.0], "std": [bad_std, 0.5]},
    }
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="non-finite or non-positive z-score std"):
        cw._load_wind_component_norm_params(path)


def test_load_wind_component_norm_params_returns_valid_params(tmp_path: Path) -> None:
    path = tmp_path / "weather_norm_params.json"
    _write_norm_params(path)

    assert cw._load_wind_component_norm_params(path) == (1.0, 2.0, -1.0, 0.5)


@pytest.mark.parametrize("bad_value", [True, False, "north", None, float("nan"), float("inf")])
def test_apply_weather_edit_rejects_non_numeric_direction_degrees(tmp_path: Path, bad_value: object) -> None:
    """`direction_degrees: true` must not be silently accepted as 1.0 degrees."""
    raw, processed = _zone_dependent_weather_frames()
    norm_params_path = tmp_path / "weather_norm_params.json"
    _write_norm_params(norm_params_path)

    with pytest.raises(ValueError, match="must define a (numeric|finite) 'direction_degrees'"):
        cw.apply_weather_edit(
            raw,
            processed,
            mode=cw.WIND_DIRECTION_ZONE_DEPENDENT_TRANSPLANT_MODE,
            scenario_name="scenario",
            recipient_hex_ids=["16"],
            params={
                "donor_hex_ids": ["16"],
                "direction_degrees": bad_value,
                "wind_speed_percentile": 90.0,
            },
            norm_params_path=norm_params_path,
        )


def test_apply_weather_edit_accepts_numeric_string_direction_degrees(tmp_path: Path) -> None:
    """YAML-quoted numbers still parse, so the stricter check does not break valid configs."""
    raw, processed = _zone_dependent_weather_frames()
    norm_params_path = tmp_path / "weather_norm_params.json"
    _write_norm_params(norm_params_path)

    _, reports = cw.apply_weather_edit(
        raw,
        processed,
        mode=cw.WIND_DIRECTION_ZONE_DEPENDENT_TRANSPLANT_MODE,
        scenario_name="scenario",
        recipient_hex_ids=["16"],
        params={
            "donor_hex_ids": ["16"],
            "direction_degrees": "45",
            "wind_speed_percentile": "90",
        },
        norm_params_path=norm_params_path,
    )

    assert reports
    for report in reports:
        assert report.direction_degrees == pytest.approx(45.0)
        assert report.wind_speed_percentile == pytest.approx(90.0)

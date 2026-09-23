"""Tests for the project input checks (python -m inference.check) on a tiny synthetic BurnP3+ project."""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from inference.bundle import ModelBundle, load_bundle
from inference.check import CheckReport, check_project, main, resolve_hex_ids
from tests.test_fuels import write_project_fuel_tables
from tests.test_predict import CRS, HEIGHT, MASK_COLS, MASK_ROWS, ORIGIN_X, ORIGIN_Y, WIDTH, _write_hexel, _write_raster, make_test_bundle


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> ModelBundle:
    return load_bundle(make_test_bundle(tmp_path_factory.mktemp("check_bundle")))


@pytest.fixture
def project(tmp_path: Path) -> Path:
    project_dir = tmp_path / "project"
    _write_hexel(project_dir)
    return project_dir


def _messages(report: CheckReport, level: str) -> list[str]:
    return [f"{finding.path}: {finding.message}" for finding in report.findings if finding.level == level]


def test_clean_project_passes_and_names_the_default_fire_size_table(bundle: ModelBundle, project: Path):
    report = check_project(bundle, project)

    assert report.ok
    assert report.hexels == ["hex01"]
    assert report.errors == []
    assert report.warnings == []
    assert len(report.notes) == 1
    assert "national training table shipped with the model (df_fire_fru_training.csv, 6 fires)" in report.notes[0].message
    text = report.format()
    assert "NOTE" in text
    assert "hex01: OK" in text
    assert "ready to predict, no problems found" in text


def test_user_fire_size_table_is_named_and_validated(bundle: ModelBundle, project: Path, tmp_path: Path):
    table = tmp_path / "regional.csv"
    pd.DataFrame({"GRIDCODE": [21, 21], "Fsize": [100.0, 2000.0]}).to_csv(table, index=False)

    report = check_project(bundle, project, fire_size_table=table)

    assert report.ok
    assert f"using your table {table}" in report.notes[0].message
    assert _messages(report, "warning") == [
        "hex01/spatial/hex01_firezones.tif: Fire zone(s) 22 are not in the fire-size table; "
        "the model uses the fire-size distribution of the whole table there."
    ]

    pd.DataFrame({"GRIDCODE": [21], "HECTARES": [100.0]}).to_csv(table, index=False)
    report = check_project(bundle, project, fire_size_table=table)
    assert not report.ok
    assert report.errors[0].hexel is None
    assert "missing column(s) ['SIZE_HA']" in report.errors[0].message


def test_unknown_fuel_codes_are_listed_with_cell_counts(bundle: ModelBundle, project: Path):
    cols = np.mgrid[0:HEIGHT, 0:WIDTH][1]
    _write_raster(project / "hex01" / "spatial" / "hex01_fbp.tif", np.where(cols < WIDTH // 2, 1, 999).astype(np.int16), -9999)

    report = check_project(bundle, project)

    cells_inside_mask = (MASK_ROWS[1] - MASK_ROWS[0]) * (MASK_COLS[1] - WIDTH // 2)
    assert _messages(report, "error") == [
        f"hex01/spatial/hex01_fbp.tif: Fuel code(s) the model has no fuel curve for: 999 ({cells_inside_mask:,} cells): "
        "not in the model's fuel table and no hex01_FuelTypes.csv / hex01_FuelCodeCrosswalk.csv to define it. "
        "Known FBP codes: 1, 13. Recode these cells, declare them nodata, or define them in the project's fuel tables."
    ]


def test_fuel_codes_defined_in_the_project_fuel_tables_get_derived_curves(bundle: ModelBundle, project: Path):
    rows, cols = np.mgrid[0:HEIGHT, 0:WIDTH]
    fuel = np.where(cols < WIDTH // 2, np.where(rows < HEIGHT // 2, 1, 425), np.where(rows < HEIGHT // 2, 13, 21))
    _write_raster(project / "hex01" / "spatial" / "hex01_fbp.tif", fuel.astype(np.int16), -9999)
    write_project_fuel_tables(
        project,
        "01",
        {
            425: ("Boreal Mixedwood - Leafless (25% Conifer)", "M-1 (25 PC)"),
            21: ("Jack or Lodgepole Pine Slash", "S-1"),
        },
    )

    report = check_project(bundle, project)

    quarter = (HEIGHT // 2 - MASK_ROWS[0]) * (WIDTH // 2 - MASK_COLS[0])
    assert _messages(report, "error") == [
        f"hex01/spatial/hex01_fbp.tif: Fuel code(s) the model has no fuel curve for: 21 ({quarter:,} cells): "
        "S-1: fuel type S-1 needs curves from the R script. "
        "Known FBP codes: 1, 13. Recode these cells, declare them nodata, or define them in the project's fuel tables."
    ]
    assert (
        "hex01/tabular/hex01_FuelCodeCrosswalk.csv: Fuel code(s) not in the model's fuel table, with curves computed from "
        f"the project's fuel tables and the FBP equations: 425 = M-1 (25 PC) ({quarter:,} cells)."
    ) in _messages(report, "note")
    assert _messages(report, "warning") == []


def test_missing_files_are_all_reported(bundle: ModelBundle, project: Path):
    for name in ("spatial/hex01_dem.tif", "tabular/hex01_GreenUp.csv", "tabular/hex01_DailyWeather.csv"):
        (project / "hex01" / name).unlink()

    report = check_project(bundle, project)

    assert _messages(report, "error") == [
        "hex01/spatial/hex01_dem.tif: Missing DEM.",
        "hex01/tabular/hex01_GreenUp.csv: Missing green-up table (Season, GreenUp).",
        "hex01/tabular/hex01_DailyWeather.csv: Missing daily weather table.",
    ]


def test_dem_far_from_the_training_resolution_is_an_error(bundle: ModelBundle, project: Path):
    dem = project / "hex01" / "spatial" / "hex01_dem.tif"
    profile = {"driver": "GTiff", "height": 16, "width": 20, "count": 1, "dtype": "float32", "crs": CRS, "nodata": -9999.0}
    with rasterio.open(dem, "w", transform=from_origin(ORIGIN_X, ORIGIN_Y, 250.0, 250.0), **profile) as dst:
        dst.write(np.full((16, 20), 300.0, dtype=np.float32), 1)

    report = check_project(bundle, project)

    errors = _messages(report, "error")
    assert len(errors) == 1
    assert "DEM cells are 250.0 m" in errors[0]
    assert "Resample the inputs to 100 m" in errors[0]
    assert any("differs from the DEM (250.0 m)" in message for message in _messages(report, "warning"))


def _rewrite_weather_zones(project: Path, mapping: dict[str, str | None]) -> None:
    path = project / "hex01" / "tabular" / "hex01_DailyWeather.csv"
    weather = pd.read_csv(path)
    weather["WeatherZone"] = weather["WeatherZone"].map(lambda zone: mapping.get(zone, zone))
    weather.dropna(subset=["WeatherZone"]).to_csv(path, index=False)


def test_weather_zones_that_match_no_fire_zone_are_an_error(bundle: ModelBundle, project: Path):
    _rewrite_weather_zones(project, {"fru21": "fru31", "fru22": "fru32"})

    report = check_project(bundle, project)

    assert len(report.errors) == 1
    assert "None of the fire zones in the fire-zone raster (21, 22) have weather rows (weather zones: 31, 32)" in report.errors[0].message


def test_fire_zone_without_weather_is_a_warning(bundle: ModelBundle, project: Path):
    _rewrite_weather_zones(project, {"fru22": None})

    report = check_project(bundle, project)

    assert report.ok
    assert _messages(report, "warning") == [
        "hex01/tabular/hex01_DailyWeather.csv: Fire zone(s) 22 have no weather rows; the model uses this hexel's average weather there."
    ]


def test_implausible_weather_values_are_warnings(bundle: ModelBundle, project: Path):
    path = project / "hex01" / "tabular" / "hex01_DailyWeather.csv"
    weather = pd.read_csv(path)
    weather.loc[0, "RelativeHumidity"] = 150
    weather.to_csv(path, index=False)

    report = check_project(bundle, project)

    assert report.ok
    assert _messages(report, "warning") == [
        "hex01/tabular/hex01_DailyWeather.csv: RelativeHumidity: 1 value(s) outside the plausible range [0.0, 100.0]."
    ]


def test_raster_without_crs_and_mask_outside_the_rasters(bundle: ModelBundle, project: Path):
    spatial = project / "hex01" / "spatial"
    with rasterio.open(
        spatial / "hex01_fbp.tif",
        "w",
        driver="GTiff",
        height=HEIGHT,
        width=WIDTH,
        count=1,
        dtype="int16",
        transform=from_origin(ORIGIN_X, ORIGIN_Y, 100.0, 100.0),
    ) as dst:
        dst.write(np.ones((HEIGHT, WIDTH), dtype=np.int16), 1)
    far_away = box(ORIGIN_X + 1e6, ORIGIN_Y - 1e6, ORIGIN_X + 1e6 + 1000, ORIGIN_Y - 1e6 + 1000)
    gpd.GeoDataFrame({"id": [1]}, geometry=[far_away], crs=CRS).to_file(spatial / "mask_grids" / "hex01_actual.shp")

    report = check_project(bundle, project)

    errors = _messages(report, "error")
    assert "hex01/spatial/hex01_fbp.tif: Fuel raster has no coordinate reference system." in errors
    assert "hex01/spatial/hex01_dem.tif: DEM does not overlap the 'actual' mask." in errors
    assert "hex01/spatial/hex01_firezones.tif: Fire-zone raster does not overlap the 'actual' mask." in errors


def test_ignition_season_without_green_up_entry_is_an_error(bundle: ModelBundle, project: Path):
    pd.DataFrame({"Season": ["s1"], "GreenUp": ["No"]}).to_csv(project / "hex01" / "tabular" / "hex01_GreenUp.csv", index=False)

    report = check_project(bundle, project)

    assert _messages(report, "error") == [
        "hex01/tabular/hex01_GreenUp.csv: Season(s) s2 of the ignition distribution table have no green-up entry; "
        "the green-up table lists s1."
    ]


def test_project_without_usable_hexel_folders(bundle: ModelBundle, tmp_path: Path):
    (tmp_path / "empty").mkdir()
    assert "No hexel folders found" in check_project(bundle, tmp_path / "empty").errors[0].message

    (tmp_path / "named" / "hexA").mkdir(parents=True)
    report = check_project(bundle, tmp_path / "named")
    assert _messages(report, "error") == ["hexA: Hexel folders must be named 'hex' followed by digits, e.g. hex07."]


def test_resolve_hex_ids_accepts_unpadded_and_prefixed_ids():
    assert resolve_hex_ids(["1", "hex12", "012", "7"], ["01", "12"]) == (["01", "12", "12"], ["7"])


def test_cli_exit_codes_and_json_report(bundle: ModelBundle, project: Path, tmp_path: Path, capsys):
    report_path = tmp_path / "report.json"
    assert main(["--bundle", str(bundle.root), "--project", str(project), "--json", str(report_path)]) == 0
    assert "ready to predict" in capsys.readouterr().out
    assert json.loads(report_path.read_text())["ok"] is True

    (project / "hex01" / "spatial" / "hex01_dem.tif").unlink()
    assert main(["--bundle", str(bundle.root), "--project", str(project), "--json", str(report_path)]) == 1
    assert "Fix the errors before predicting" in capsys.readouterr().out
    assert json.loads(report_path.read_text())["num_errors"] == 1

    assert main(["--bundle", str(tmp_path / "no-bundle"), "--project", str(project)]) == 2
    assert "Model bundle directory not found" in capsys.readouterr().err


def test_outputs_mode_checks_the_burnp3_results_for_evaluation(bundle: ModelBundle, project: Path):
    from tests.test_evaluate import _write_burnp3_results

    report = check_project(bundle, project, outputs=True)

    assert not report.ok
    assert _messages(report, "error") == [
        "hex01/results/burnP3Plus_OutputBurnProbability/burnProbability-sn2.tif: Missing BurnP3+ burn probability output.",
        "hex01/results/burnP3Plus_OutputFireIntensitySummaryMap/fbpSummary-FireIntensity-Average.tif: "
        "Missing BurnP3+ fire intensity output.",
        "hex01/results/burnP3Plus_OutputRateOfSpreadSummaryMap/fbpSummary-RateOfSpread-Average.tif: Missing BurnP3+ rate of spread output.",
    ]
    assert "Fix the errors before evaluating." in report.format()

    _write_burnp3_results(project)
    report = check_project(bundle, project, outputs=True, inputs=False)

    assert report.ok
    assert report.notes == []  # the fire-size table is an input
    assert "ready to evaluate, no problems found" in report.format()


@pytest.mark.parametrize("mask_scope", ["actual", "none"])
def test_burnp3_outputs_without_data_are_errors_but_partial_nodata_is_normal(bundle: ModelBundle, project: Path, mask_scope: str):
    from data_preparation.paths import Paths
    from tests.test_evaluate import _write_burnp3_results

    grids = _write_burnp3_results(project)
    paths = Paths(hex_id="01", root_dir=project)
    _write_raster(paths.output_fire_intensity(), np.full((HEIGHT, WIDTH), -9999.0, dtype=np.float32), -9999.0)
    _write_raster(paths.output_ros(), np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32), -9999.0)
    bp = grids["bp"].copy()
    bp[:, : WIDTH // 2] = -9999.0  # nothing burned there
    _write_raster(paths.output_burn_prob(), bp, -9999.0)

    report = check_project(bundle, project, outputs=True, inputs=False, mask_scope=mask_scope)

    area = "" if mask_scope == "none" else " inside the 'actual' mask"
    assert _messages(report, "error") == [
        f"hex01/results/burnP3Plus_OutputFireIntensitySummaryMap/fbpSummary-FireIntensity-Average.tif: "
        f"BurnP3+ fire intensity output has no valid data{area}.",
        f"hex01/results/burnP3Plus_OutputRateOfSpreadSummaryMap/fbpSummary-RateOfSpread-Average.tif: "
        f"BurnP3+ rate of spread output has no valid data{area}.",
    ]
    assert report.warnings == []


def test_ignition_grid_without_data_inside_the_mask_is_an_error(bundle: ModelBundle, project: Path):
    grid = project / "hex01" / "spatial" / "ignition_grids" / "hex01_ignGrid_H_s1.tif"
    assert grid.is_file()
    empty = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    empty[MASK_ROWS[0] : MASK_ROWS[1], MASK_COLS[0] : MASK_COLS[1]] = -9999.0
    _write_raster(grid, empty, -9999.0)

    report = check_project(bundle, project)

    assert _messages(report, "error") == [
        "hex01/spatial/ignition_grids/hex01_ignGrid_H_s1.tif: Ignition grid has no valid data inside the 'actual' mask."
    ]


def test_known_fuel_codes_defined_as_another_fuel_are_flagged(bundle: ModelBundle, project: Path):
    # The test bundle's curves are synthetic, so a real C-1 definition of code 1 differs from the model's curve.
    write_project_fuel_tables(project, "01", {1: ("Spruce-Lichen Woodland", "C-1"), 13: ("Plantation", "C-6")})

    report = check_project(bundle, project)

    assert report.ok
    assert _messages(report, "warning") == [
        "hex01/tabular/hex01_FuelCodeCrosswalk.csv: Fuel code(s) the project defines as a fuel whose curve cannot be "
        "computed here, so the model's curve for the code is used; check that the fuel raster uses the model's codes: "
        f"13 = C-6: fuel type C-6 needs curves from the R script ({(MASK_ROWS[1] - MASK_ROWS[0]) * (MASK_COLS[1] - WIDTH // 2):,} cells).",
        "hex01/tabular/hex01_FuelCodeCrosswalk.csv: Fuel code(s) the project defines differently from the model's fuel "
        f"table; the project's definition is used: 1 = C-1 ({(MASK_ROWS[1] - MASK_ROWS[0]) * (WIDTH // 2 - MASK_COLS[0]):,} cells). "
        "Check that the fuel raster uses the same codes.",
    ]


def test_missing_mask_is_an_error_that_points_to_mask_scope_none(bundle: ModelBundle, project: Path):
    for path in (project / "hex01" / "spatial" / "mask_grids").iterdir():
        path.unlink()

    report = check_project(bundle, project)

    errors = _messages(report, "error")
    assert len(errors) == 1
    assert errors[0].startswith("hex01/spatial/mask_grids/hex01_actual.shp: Missing the 'actual' mask shapefile")
    assert "--mask_scope none" in errors[0]


def test_regional_study_area_without_mask_checks_the_whole_raster(bundle: ModelBundle, project: Path):
    for path in (project / "hex01" / "spatial" / "mask_grids").iterdir():
        path.unlink()

    report = check_project(bundle, project, mask_scope="none")

    assert _messages(report, "error") == []
    assert _messages(report, "warning") == []
    assert report.mask_scope == "none"

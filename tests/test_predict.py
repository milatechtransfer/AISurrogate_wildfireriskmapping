"""End-to-end tests for bundle-based, input-only prediction on a tiny synthetic BurnP3+ project."""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import torch
from rasterio.transform import from_origin
from shapely.geometry import box

from inference.bundle import RESOURCE_DATASET_NORM_STATS, RESOURCE_FUEL_CURVES, load_bundle
from inference.predict import RUN_MANIFEST_FILENAME, WORK_DIRNAME, PredictError, main, run_predict
from src.config import Config
from src.datasets.sources.grids import GridSource, GridSourceResources
from tests.test_bundle import FEATURE_CHANNEL_MAP, NUM_ISI_BINS, _build, _export, _tiny_config, _write_training_resources

CRS = "ESRI:102002"
CELL = 100.0
HEIGHT, WIDTH = 40, 48
ORIGIN_X, ORIGIN_Y = 1_000_000.0, 1_000_000.0
WEATHER_ZONES = {21: "fru21", 22: "fru22"}
MASK_ROWS = (4, 36)
MASK_COLS = (4, 44)


def _write_full_weather_norm_params(data_root: Path) -> None:
    z_cols = [
        "Temperature",
        "WindSpeed",
        "DuffMoistureCode",
        "DroughtCode",
        "InitialSpreadIndex",
        "BuildupIndex",
        "FireWeatherIndex",
        "wind_x",
        "wind_y",
    ]
    params = {
        "min_max": {"cols": ["RelativeHumidity", "FineFuelMoistureCode"], "min": [0.0, 77.7], "max": [100.0, 99.0]},
        "z_score": {"cols": z_cols, "mean": [20.0] * len(z_cols), "std": [5.0] * len(z_cols)},
    }
    (data_root / "weather_norm_params.json").write_text(json.dumps(params))


@pytest.fixture
def bundle_dir(tmp_path: Path) -> Path:
    data_root = tmp_path / "training" / "data_samples"
    _write_training_resources(data_root)
    _write_full_weather_norm_params(data_root)
    config = Config(**_tiny_config(data_root))
    torch.manual_seed(0)
    model = _build(config, spatial_channels=20, fuel_curve_len=NUM_ISI_BINS).eval()
    checkpoint_path = tmp_path / "training" / "run" / "best.pth"
    checkpoint_path.parent.mkdir(parents=True)
    torch.save({"model_state": model.state_dict(), "epoch": 1, "config": config.model_dump()}, checkpoint_path)
    denominator_json = tmp_path / "training" / "hazard_scale_denominator.json"
    denominator_json.write_text(json.dumps({"scale_denominator": 50.0, "scale_denominator_source": "test"}))
    return _export(checkpoint_path, tmp_path / "bundle", hazard_denominator_json=denominator_json)


def _write_raster(path: Path, data: np.ndarray, nodata: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": data.dtype.name,
        "crs": CRS,
        "transform": from_origin(ORIGIN_X, ORIGIN_Y, CELL, CELL),
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def _write_hexel(project: Path, hex_id: str = "01", green_up_s2: str = "Yes") -> Path:
    hexel = project / f"hex{hex_id}"
    spatial, tabular = hexel / "spatial", hexel / "tabular"
    rows, cols = np.mgrid[0:HEIGHT, 0:WIDTH]
    _write_raster(spatial / f"hex{hex_id}_dem.tif", (200.0 + 5.0 * rows + 2.0 * cols).astype(np.float32), -9999.0)
    _write_raster(spatial / f"hex{hex_id}_fbp.tif", np.where(cols < WIDTH // 2, 1, 13).astype(np.int16), -9999)
    _write_raster(spatial / f"hex{hex_id}_firezones.tif", np.where(rows < HEIGHT // 2, 21, 22).astype(np.int16), -9999)
    for cause in ("H", "N"):
        for season_index, season in enumerate(("s1", "s2")):
            grid = (0.01 * (1 + season_index) + 0.001 * cols).astype(np.float32)
            _write_raster(spatial / "ignition_grids" / f"hex{hex_id}_ignGrid_{cause}_{season}.tif", grid, -9999.0)
    mask_polygon = box(
        ORIGIN_X + MASK_COLS[0] * CELL,
        ORIGIN_Y - MASK_ROWS[1] * CELL,
        ORIGIN_X + MASK_COLS[1] * CELL,
        ORIGIN_Y - MASK_ROWS[0] * CELL,
    )
    (spatial / "mask_grids").mkdir(parents=True)
    gpd.GeoDataFrame({"id": [1]}, geometry=[mask_polygon], crs=CRS).to_file(spatial / "mask_grids" / f"hex{hex_id}_actual.shp")

    tabular.mkdir(parents=True)
    weather_rows = []
    for zone_name in WEATHER_ZONES.values():
        for season in ("s1", "s2"):
            for day in range(3):
                weather_rows.append(
                    {
                        "Order": day + 1,
                        "Season": season,
                        "WeatherZone": zone_name,
                        "Temperature": 18.0 + day,
                        "RelativeHumidity": 30 + day,
                        "WindSpeed": 15 + day,
                        "WindDirection": 180 + 20 * day,
                        "Precipitation": 0.0,
                        "FineFuelMoistureCode": 90.0,
                        "DuffMoistureCode": 40.0,
                        "DroughtCode": 300.0,
                        "InitialSpreadIndex": 10.0 + day,
                        "BuildupIndex": 50.0,
                        "FireWeatherIndex": 20.0,
                    }
                )
    pd.DataFrame(weather_rows).to_csv(tabular / f"hex{hex_id}_DailyWeather.csv", index=False)
    pd.DataFrame({"Name": list(WEATHER_ZONES.values()), "ID": list(WEATHER_ZONES)}).to_csv(
        tabular / f"hex{hex_id}_FireZones.csv", index=False
    )
    distribution = [
        {"Iteration": "", "Timestep": "", "Season": season, "Cause": cause, "FireZone": zone, "RelativeLikelihood": weight}
        for season, weight in (("s1", 1.0), ("s2", 3.0))
        for cause in ("Human", "Lightning")
        for zone in WEATHER_ZONES.values()
    ]
    pd.DataFrame(distribution).to_csv(tabular / f"hex{hex_id}_IgnitionDistribution.csv", index=False)
    pd.DataFrame({"Season": ["s1", "s2"], "GreenUp": ["No", green_up_s2]}).to_csv(tabular / f"hex{hex_id}_GreenUp.csv", index=False)
    return hexel


def _write_fire_size_table(path: Path) -> Path:
    pd.DataFrame({"GRIDCODE": [21, 21, 21, 22, 22, 22], "SIZE_HA": [30.0, 120.0, 900.0, 50.0, 400.0, 5000.0]}).to_csv(path, index=False)
    return path


@pytest.fixture
def project(tmp_path: Path) -> Path:
    project_dir = tmp_path / "project"
    _write_hexel(project_dir)
    return project_dir


def test_predict_writes_georeferenced_outputs_without_burnp3_results(bundle_dir: Path, project: Path, tmp_path: Path):
    output_dir = tmp_path / "out"
    assert not (project / "hex01" / "results").exists()

    run = run_predict(
        bundle_dir=bundle_dir,
        project_dir=project,
        output_dir=output_dir,
        fire_size_table=_write_fire_size_table(tmp_path / "fire_sizes.csv"),
        device="cpu",
    )

    assert [result.hex_id for result in run.hexels] == ["01"]
    expected = {"bp", "fi", "ros", "hazard_raw", "hazard_scaled", "hazard_class"}
    assert set(run.hexels[0].outputs) == expected
    assert not (output_dir / WORK_DIRNAME).exists()

    bundle = load_bundle(bundle_dir)
    bp_max = bundle.target("bp").max_value
    assert bp_max is not None
    with rasterio.open(output_dir / "hex01" / "hex01_bp.tif") as src:
        bp = src.read(1, masked=True)
        assert src.crs.to_string() == CRS or src.crs == rasterio.crs.CRS.from_user_input(CRS)
        assert src.res == (CELL, CELL)
        assert src.dtypes[0] == "float32"
        assert src.tags(1)["units"] == "probability (0-1)"
    # Predictions cover the masked hexel area and stay inside the training BP range.
    assert bp.count() == (MASK_ROWS[1] - MASK_ROWS[0]) * (MASK_COLS[1] - MASK_COLS[0])
    assert float(bp.min()) >= 0.0
    assert float(bp.max()) <= bp_max
    with rasterio.open(output_dir / "hex01" / "hex01_fi.tif") as src:
        assert src.read(1, masked=True).min() > 0.0
    with rasterio.open(output_dir / "hex01" / "hex01_hazard_class.tif") as src:
        classes = src.read(1, masked=True)
        assert src.dtypes[0] == "uint8"
        assert classes.count() == bp.count()

    run_manifest = json.loads((output_dir / RUN_MANIFEST_FILENAME).read_text())
    assert run_manifest["status"] == "success"
    assert run_manifest["bundle"]["name"] == "test-bundle"
    assert run_manifest["options"]["device"] == "cpu"
    assert run_manifest["fire_size_table"]["matches_training_table"] is False
    assert (output_dir / "predict.log").is_file()


def test_predict_requires_fire_size_table(bundle_dir: Path, project: Path, tmp_path: Path):
    with pytest.raises(PredictError, match="fire_size_table"):
        run_predict(bundle_dir=bundle_dir, project_dir=project, output_dir=tmp_path / "out", fire_size_table=None, device="cpu")
    run_manifest = json.loads((tmp_path / "out" / RUN_MANIFEST_FILENAME).read_text())
    assert run_manifest["status"] == "failed"


def test_predict_refuses_non_empty_output_without_overwrite(bundle_dir: Path, project: Path, tmp_path: Path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "something.txt").write_text("keep me")
    with pytest.raises(PredictError, match="--overwrite"):
        run_predict(bundle_dir=bundle_dir, project_dir=project, output_dir=output_dir, fire_size_table=None, device="cpu")
    assert (output_dir / "something.txt").read_text() == "keep me"


def test_predict_reports_unknown_hexel(bundle_dir: Path, project: Path, tmp_path: Path):
    with pytest.raises(PredictError, match=r"\['07'\] not found"):
        run_predict(bundle_dir=bundle_dir, project_dir=project, output_dir=tmp_path / "out", fire_size_table=None, hex_ids=["hex07"])


def test_predict_cli_prints_plain_error(bundle_dir: Path, tmp_path: Path, capsys):
    code = main(["--bundle", str(bundle_dir), "--project", str(tmp_path / "missing"), "--output", str(tmp_path / "out")])
    assert code == 2
    assert "Project folder not found" in capsys.readouterr().err


def _write_season_dependent_fuel_curves(path: Path) -> Path:
    rows = [
        {"fbp_code": code, "SeasonState": state, "ISI": isi, "ROS": ros, "HFI": ros}
        for code, states in ((1, {"direct": 1.0}), (13, {"green": 2.0, "leafless": 8.0}))
        for state, scale in states.items()
        for isi in range(NUM_ISI_BINS)
        for ros in [scale * (isi + 1)]
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _grid_source(bundle_dir: Path, project: Path, norm_stats: dict | None = None, fuel_curves_path: Path | None = None) -> GridSource:
    bundle = load_bundle(bundle_dir)
    return GridSource(
        root_dir=str(project / "does-not-exist"),
        params=bundle.manifest.grid_params(),
        resources=GridSourceResources(
            norm_stats=norm_stats if norm_stats is not None else bundle.read_json_resource(RESOURCE_DATASET_NORM_STATS),
            channel_feature_map=FEATURE_CHANNEL_MAP,
            fuel_curves_path=fuel_curves_path or bundle.resource_path(RESOURCE_FUEL_CURVES),
            season_data_dir=project,
            hex_ids=["01"],
        ),
    )


def test_grid_source_resources_blend_seasons_from_the_users_project(bundle_dir: Path, tmp_path: Path):
    green_project, leafless_project = tmp_path / "green", tmp_path / "leafless"
    _write_hexel(green_project, green_up_s2="Yes")
    _write_hexel(leafless_project, green_up_s2="No")

    curves = _write_season_dependent_fuel_curves(tmp_path / "curves.csv")

    green = _grid_source(bundle_dir, green_project, fuel_curves_path=curves)
    leafless = _grid_source(bundle_dir, leafless_project, fuel_curves_path=curves)

    # Ignition weights are s1: 1, s2: 3. With s2 green-up, code 13 blends 3/4 green + 1/4 leafless;
    # without green-up both seasons are leafless. Code 1 has a single season state and is unaffected.
    isi = np.arange(1, NUM_ISI_BINS + 1, dtype=np.float32)
    np.testing.assert_allclose(green._dense_fuel_per_hex["01"][13], (0.75 * 2.0 + 0.25 * 8.0) * isi, rtol=1e-6)
    np.testing.assert_allclose(leafless._dense_fuel_per_hex["01"][13], 8.0 * isi, rtol=1e-6)
    np.testing.assert_array_equal(green._dense_fuel_per_hex["01"][1], leafless._dense_fuel_per_hex["01"][1])
    assert green.ELEVATION_MIN == -10.0
    assert green.ELEVATION_MAX == 3000.0


def test_grid_source_resources_fail_on_incomplete_norm_stats(bundle_dir: Path, project: Path):
    bundle = load_bundle(bundle_dir)
    norm_stats = bundle.read_json_resource(RESOURCE_DATASET_NORM_STATS)
    del norm_stats["elevation"]
    with pytest.raises(ValueError, match="'elevation'"):
        _grid_source(bundle_dir, project, norm_stats=norm_stats)


def test_grid_source_resources_reject_hexel_without_season_weights(bundle_dir: Path, project: Path):
    source = _grid_source(bundle_dir, project)
    patch = np.full((4, 4, 8), np.nan, dtype=np.float32)
    patch[..., :5] = 1.0
    patch[..., 0] = 13.0
    with pytest.raises(ValueError, match="no season weights"):
        source.get_sample({"data": patch, "hex_id": 2})

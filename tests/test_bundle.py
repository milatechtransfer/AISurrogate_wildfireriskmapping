import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from inference.bundle import (
    MANIFEST_FILENAME,
    RESOURCE_FIRE_SIZE_TABLE,
    BundleError,
    compute_model_io_dims,
    load_bundle,
)
from inference.export_bundle import export_bundle
from src.config import Config
from src.models.factory import build_model

REPO_ROOT = Path(__file__).resolve().parents[1]
Q3_CONFIG = REPO_ROOT / "configs" / "multi_output_spatial_weather_firesize_q3.yaml"
NUM_ISI_BINS = 5
FUEL_CURVE_LOG_MEAN = 2.5
FUEL_CURVE_LOG_STD = 1.25
FEATURE_CHANNEL_MAP = {
    "fuel_grid": [0],
    "elevation_grid": [1],
    "ignition_grid_human": [2],
    "ignition_grid_lightning": [3],
    "firezones_grid": [4],
    "bp_out_grid": [5],
    "fi_out_grid": [6],
    "ros_out_grid": [7],
}


def _write_training_resources(data_root: Path) -> None:
    data_root.mkdir(parents=True)
    norm_stats = {
        "elevation": {"min": -10.0, "max": 3000.0},
        "fuel_curve_iROS": {"log_mean": FUEL_CURVE_LOG_MEAN, "log_std": FUEL_CURVE_LOG_STD},
        "fire_intensity": {"log_mean": 7.9, "log_std": 1.1},
        "fire_ros": {"log_mean": 1.8, "log_std": 0.55},
        "fire_burn_probability": {"min": 1e-5, "max": 0.12},
    }
    (data_root / "dataset_norm_stats.json").write_text(json.dumps(norm_stats))
    (data_root / "weather_norm_params.json").write_text(
        json.dumps(
            {"min_max": {"cols": ["RelativeHumidity"], "min": [0.0], "max": [100.0]}, "z_score": {"cols": [], "mean": [], "std": []}}
        )
    )
    (data_root / "fire_size_norm_params.json").write_text(json.dumps({"log_size_min": 0.0, "log_size_max": 6.0}))
    (data_root / "feature_channel_map_1.json").write_text(json.dumps(FEATURE_CHANNEL_MAP))
    rows = [
        {"fbp_code": code, "SeasonState": state, "ISI": isi, "ROS": float(code + isi), "HFI": float(code * isi)}
        for code, states in ((1, ["direct"]), (13, ["green", "leafless"]))
        for state in states
        for isi in range(NUM_ISI_BINS)
    ]
    pd.DataFrame(rows).to_csv(data_root / "fbp_curves_national_fuel.csv", index=False)


def _tiny_config(data_root: Path) -> dict:
    config = yaml.safe_load(Q3_CONFIG.read_text())
    config["model"]["hidden_features"] = [4, 8]
    config["data"]["root_dir"] = str(data_root)
    config["data"]["raw_data_dir"] = str(data_root.parent)
    return config


def _build(config: Config, spatial_channels: int, fuel_curve_len: int) -> torch.nn.Module:
    return build_model(
        model_config=config.model,
        spatial_input_channels=spatial_channels,
        auxiliary_input_dims={"fuel_curve": fuel_curve_len},
        fuel_curve_mean=torch.tensor([FUEL_CURVE_LOG_MEAN]),
        fuel_curve_std=torch.tensor([FUEL_CURVE_LOG_STD]),
        target_names=["bp", "fi", "ros"],
    )


@pytest.fixture
def training_checkpoint(tmp_path: Path) -> tuple[Path, torch.nn.Module]:
    data_root = tmp_path / "data_samples"
    _write_training_resources(data_root)
    config_dict = _tiny_config(data_root)
    config = Config(**config_dict)
    torch.manual_seed(0)
    model = _build(config, spatial_channels=20, fuel_curve_len=NUM_ISI_BINS).eval()
    checkpoint_path = tmp_path / "run" / "best.pth"
    checkpoint_path.parent.mkdir()
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": {},
            "epoch": 3,
            "metric_value": {"bp/ccc": 0.5},
            "config": config.model_dump(),
        },
        checkpoint_path,
    )
    pd.DataFrame([{"val_hexel/all/bp_ccc": 0.8, "val_hexel/all/fi_ccc": 0.7}]).to_csv(checkpoint_path.parent / "val_results.csv")
    return checkpoint_path, model


def _write_training_fire_size_table(path: Path) -> Path:
    pd.DataFrame({"GRIDCODE": [21, 21, 21, 22, 22, 22], "SIZE_HA": [30.0, 120.0, 900.0, 50.0, 400.0, 5000.0]}).to_csv(path, index=False)
    return path


def _export(checkpoint_path: Path, out_dir: Path, **kwargs) -> Path:
    if "fire_size_table" not in kwargs:
        kwargs["fire_size_table"] = _write_training_fire_size_table(checkpoint_path.parent / "df_fire_fru_training.csv")
    return export_bundle(checkpoint_path=checkpoint_path, out_dir=out_dir, name="test-bundle", version="0.0.1", **kwargs)


def test_compute_model_io_dims_for_q3_config(tmp_path: Path):
    data_root = tmp_path / "data"
    _write_training_resources(data_root)
    config = Config(**_tiny_config(data_root))

    spatial, auxiliary = compute_model_io_dims(config.data.input_sources, FEATURE_CHANNEL_MAP, data_root / "fbp_curves_national_fuel.csv")

    # ignition H + L + elevation (fuel moves to the iROS branch) + 3 terrain + 11 weather + 3 fire-size quantiles
    assert spatial == 3 + 3 + 11 + 3
    assert auxiliary == {"fuel_curve": NUM_ISI_BINS}


def test_export_and_load_round_trip_reproduces_model(training_checkpoint, tmp_path: Path):
    checkpoint_path, original_model = training_checkpoint
    bundle_dir = _export(checkpoint_path, tmp_path / "bundle", selection_note="test")

    bundle = load_bundle(bundle_dir)
    manifest = bundle.manifest

    assert manifest.target_names == ["bp", "fi", "ros"]
    assert bundle.target("bp").max_value == pytest.approx(0.12)
    assert bundle.target("bp").min_value == 0.0  # bp_nodata_as_zero
    assert bundle.target("fi").log_mean == pytest.approx(7.9)
    assert bundle.target("fi").units == "kW/m"
    assert manifest.model_io.spatial_input_channels == 20
    assert manifest.provenance.epoch == 3
    assert manifest.provenance.metrics["validation"]["bp_ccc"] == pytest.approx(0.8)
    assert bundle.read_json_resource("dataset_norm_stats")["elevation"]["max"] == 3000.0
    assert bundle.resource_path("fuel_curves").name == "fbp_curves_national_fuel.csv"
    assert (bundle_dir / "MODEL_CARD.md").is_file()
    assert "model.pt" in (bundle_dir / "SHA256SUMS").read_text()
    assert str(tmp_path) not in json.dumps(manifest.data.model_dump(mode="json"))

    torch.manual_seed(1)
    spatial = torch.randn(1, 20, 32, 32)
    auxiliary = {"fuel_curve": torch.rand(1, NUM_ISI_BINS, 32, 32)}
    with torch.no_grad():
        expected = original_model(spatial, auxiliary)
        actual = bundle.build_model("cpu")(spatial, auxiliary)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_export_ships_the_training_fire_size_table(training_checkpoint, tmp_path: Path):
    checkpoint_path, _ = training_checkpoint
    table = tmp_path / "df_fire_fru_25ha.csv"
    pd.DataFrame({"SIZE_HA": [1.0, 2.0], "GRIDCODE": [1, 2]}).to_csv(table)

    bundle = load_bundle(_export(checkpoint_path, tmp_path / "bundle", fire_size_table=table, fire_size_table_note="Public data."))

    recorded = bundle.manifest.inputs.fire_size_training_table
    assert recorded is not None
    assert (recorded.filename, recorded.num_rows, recorded.columns) == ("df_fire_fru_25ha.csv", 2, ["SIZE_HA", "GRIDCODE"])
    shipped = bundle.resource_path(RESOURCE_FIRE_SIZE_TABLE)
    assert shipped == bundle.root / "lookups" / "df_fire_fru_25ha.csv"
    assert shipped.read_bytes() == table.read_bytes()
    assert recorded.sha256 == bundle.manifest.resources[RESOURCE_FIRE_SIZE_TABLE].sha256
    assert "Public data." in (bundle.root / "MODEL_CARD.md").read_text()


def test_export_requires_fire_size_table_for_fire_size_models(training_checkpoint, tmp_path: Path):
    checkpoint_path, _ = training_checkpoint
    with pytest.raises(BundleError, match="--fire_size_table"):
        _export(checkpoint_path, tmp_path / "bundle", fire_size_table=None)
    assert not (tmp_path / "bundle").exists()


def test_export_refuses_existing_output_without_overwrite(training_checkpoint, tmp_path: Path):
    checkpoint_path, _ = training_checkpoint
    out_dir = tmp_path / "bundle"
    out_dir.mkdir()

    with pytest.raises(BundleError, match="already exists"):
        _export(checkpoint_path, out_dir)


def test_export_rejects_norm_stats_from_another_model(training_checkpoint, tmp_path: Path):
    checkpoint_path, _ = training_checkpoint
    other_stats = tmp_path / "other_stats.json"
    stats = json.loads((tmp_path / "data_samples" / "dataset_norm_stats.json").read_text())
    stats["fuel_curve_iROS"]["log_mean"] = 9.0
    other_stats.write_text(json.dumps(stats))

    with pytest.raises(BundleError, match="does not belong to this checkpoint"):
        _export(checkpoint_path, tmp_path / "bundle", dataset_norm_stats_path=other_stats)


def test_export_reports_missing_resources(training_checkpoint, tmp_path: Path):
    checkpoint_path, _ = training_checkpoint
    (tmp_path / "data_samples" / "weather_norm_params.json").unlink()

    with pytest.raises(BundleError, match="weather_norm_params.json"):
        _export(checkpoint_path, tmp_path / "bundle")


@pytest.fixture
def bundle_dir(training_checkpoint, tmp_path: Path) -> Path:
    checkpoint_path, _ = training_checkpoint
    return _export(checkpoint_path, tmp_path / "bundle")


def test_load_bundle_detects_corrupted_weights(bundle_dir: Path):
    with open(bundle_dir / "model.pt", "ab") as handle:
        handle.write(b"corruption")

    with pytest.raises(BundleError, match="checksum mismatch for model.pt"):
        load_bundle(bundle_dir)
    load_bundle(bundle_dir, verify_checksums=False)


def test_load_bundle_detects_missing_resource(bundle_dir: Path):
    (bundle_dir / "norm" / "weather_norm_params.json").unlink()

    with pytest.raises(BundleError, match="incomplete"):
        load_bundle(bundle_dir)


def test_load_bundle_rejects_unsupported_format_version(bundle_dir: Path):
    manifest = yaml.safe_load((bundle_dir / MANIFEST_FILENAME).read_text())
    manifest["bundle_format_version"] = 999
    (bundle_dir / MANIFEST_FILENAME).write_text(yaml.safe_dump(manifest))

    with pytest.raises(BundleError, match="not supported"):
        load_bundle(bundle_dir)


def test_load_bundle_rejects_paths_outside_bundle(bundle_dir: Path):
    manifest = yaml.safe_load((bundle_dir / MANIFEST_FILENAME).read_text())
    manifest["resources"]["fuel_curves"]["path"] = "../outside.csv"
    (bundle_dir / MANIFEST_FILENAME).write_text(yaml.safe_dump(manifest))

    with pytest.raises(BundleError, match="outside the bundle"):
        load_bundle(bundle_dir)


def test_load_bundle_rejects_non_bundle_directory(tmp_path: Path):
    with pytest.raises(BundleError, match="not a model bundle"):
        load_bundle(tmp_path)

"""Tests for evaluating a bundle against BurnP3+ outputs (python -m inference.evaluate) on a tiny synthetic project."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_preparation.paths import Paths
from inference.bundle import ModelBundle, load_bundle
from inference.evaluate import (
    EVALUATION_MANIFEST_FILENAME,
    HAZARD_PER_HEXEL_FILENAME,
    METRICS_PER_FIREZONE_FILENAME,
    METRICS_PER_HEXEL_FILENAME,
    METRICS_SUMMARY_FILENAME,
    PREDICTIONS_DIRNAME,
    EvaluateError,
    main,
    run_evaluate,
)
from inference.predict import RUN_MANIFEST_FILENAME, write_geotiff
from tests.test_predict import (
    CELL,
    CRS,
    HEIGHT,
    MASK_COLS,
    MASK_ROWS,
    ORIGIN_X,
    ORIGIN_Y,
    WIDTH,
    _write_hexel,
    _write_raster,
    make_test_bundle,
)

MASK_PIXELS = (MASK_ROWS[1] - MASK_ROWS[0]) * (MASK_COLS[1] - MASK_COLS[0])


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return make_test_bundle(tmp_path_factory.mktemp("evaluate_bundle"))


@pytest.fixture(scope="module")
def bundle(bundle_dir: Path) -> ModelBundle:
    return load_bundle(bundle_dir)


def _burnp3_grids() -> dict[str, np.ndarray]:
    rows, cols = np.mgrid[0:HEIGHT, 0:WIDTH]
    return {
        "bp": (0.001 + 0.002 * rows + 0.0005 * cols).astype(np.float32),
        "fi": (500.0 + 40.0 * cols + 10.0 * rows).astype(np.float32),
        "ros": (1.0 + 0.1 * rows).astype(np.float32),
    }


def _write_burnp3_results(project: Path, hex_id: str = "01") -> dict[str, np.ndarray]:
    grids = _burnp3_grids()
    paths = Paths(hex_id=hex_id, root_dir=project)
    for name, method in (("bp", paths.output_burn_prob), ("fi", paths.output_fire_intensity), ("ros", paths.output_ros)):
        _write_raster(method(), grids[name], -9999.0)
    return grids


def _write_predictions(
    out_dir: Path, bundle: ModelBundle, grids: dict[str, np.ndarray], hex_id: str = "01", manifest: dict | None = None
) -> Path:
    """Prediction rasters laid out like inference.predict output: cropped to the mask bounds."""
    from rasterio.transform import from_origin

    rows, cols = slice(*MASK_ROWS), slice(*MASK_COLS)
    profile = {
        "driver": "GTiff",
        "height": MASK_ROWS[1] - MASK_ROWS[0],
        "width": MASK_COLS[1] - MASK_COLS[0],
        "count": 1,
        "crs": CRS,
        "transform": from_origin(ORIGIN_X + MASK_COLS[0] * CELL, ORIGIN_Y - MASK_ROWS[0] * CELL, CELL, CELL),
    }
    (out_dir / f"hex{hex_id}").mkdir(parents=True, exist_ok=True)
    for name, grid in grids.items():
        write_geotiff(out_dir / f"hex{hex_id}" / f"hex{hex_id}_{name}.tif", grid[rows, cols], profile, description=name)
    run_manifest = {
        "bundle": {"name": bundle.manifest.name, "version": bundle.manifest.version, "weights_sha256": bundle.manifest.weights.sha256}
    }
    run_manifest["options"] = {"mask_scope": "actual"}
    run_manifest.update(manifest or {})
    (out_dir / RUN_MANIFEST_FILENAME).write_text(json.dumps(run_manifest))
    return out_dir


@pytest.fixture
def project(tmp_path: Path) -> Path:
    project_dir = tmp_path / "project"
    _write_hexel(project_dir)
    return project_dir


def test_predictions_identical_to_burnp3_score_perfectly(bundle_dir: Path, bundle: ModelBundle, project: Path, tmp_path: Path):
    grids = _write_burnp3_results(project)
    predictions = _write_predictions(tmp_path / "predictions", bundle, grids)
    output_dir = tmp_path / "evaluation"

    run = run_evaluate(bundle_dir, project, output_dir, predictions_dir=predictions, device="cpu")

    assert run.warnings == []
    assert run.input_check is not None
    assert run.input_check.ok
    metrics = pd.read_csv(output_dir / METRICS_PER_HEXEL_FILENAME)
    assert list(metrics["target"]) == ["bp", "fi", "ros", "hazard"]
    assert set(metrics["area"]) == {"actual"}
    assert set(metrics["hexel"]) == {"hex01"}
    assert (metrics["n_pixels"] == MASK_PIXELS).all()
    np.testing.assert_allclose(metrics["ccc"], 1.0, atol=1e-5)
    np.testing.assert_allclose(metrics["spearman"], 1.0, atol=1e-5)
    np.testing.assert_allclose(metrics["mae"], 0.0, atol=1e-6)
    np.testing.assert_allclose(metrics["iou_top10"], 1.0, atol=1e-6)

    summary = pd.read_csv(output_dir / METRICS_SUMMARY_FILENAME)
    assert list(summary["target"]) == ["bp", "fi", "ros", "hazard"]
    assert (summary["n_hexels"] == 1).all()

    hazard = pd.read_csv(output_dir / HAZARD_PER_HEXEL_FILENAME)
    assert hazard.loc[0, "exact_accuracy"] == pytest.approx(1.0)
    assert run.hazard_summary is not None
    assert run.hazard_summary["exact_accuracy"] == pytest.approx(1.0)
    confusion = pd.read_csv(output_dir / "hazard_confusion_matrix.csv", index_col=0)
    assert int(confusion.to_numpy().sum()) == MASK_PIXELS
    assert int(np.trace(confusion.to_numpy())) == MASK_PIXELS

    for name in ("bp", "fi", "ros"):
        for kind in ("maps", "scatter", "histogram"):
            assert (output_dir / "plots" / "hex01" / f"hex01_{name}_{kind}.png").is_file()
    assert not (output_dir / "plots" / "hex01" / "predicted_hexels_plot").exists()

    manifest = json.loads((output_dir / EVALUATION_MANIFEST_FILENAME).read_text())
    assert manifest["status"] == "success"
    assert manifest["predictions"]["made_by_this_run"] is False
    assert manifest["summary"]["bp"]["ccc"] == pytest.approx(1.0, abs=1e-5)
    assert manifest["hazard"]["scale_denominator"] == 50.0
    assert manifest["input_check"] == {"errors": [], "warnings": []}


def test_bp_missing_in_burnp3_counts_as_zero(bundle_dir: Path, bundle: ModelBundle, project: Path, tmp_path: Path):
    grids = _write_burnp3_results(project)
    bp = grids["bp"].copy()
    bp[MASK_ROWS[0], MASK_COLS[0] : MASK_COLS[1]] = -9999.0  # BurnP3+ writes nodata where nothing burned
    _write_raster(Paths(hex_id="01", root_dir=project).output_burn_prob(), bp, -9999.0)
    predictions = _write_predictions(tmp_path / "predictions", bundle, {**grids, "bp": np.where(bp < 0, 0.0, bp).astype(np.float32)})

    run = run_evaluate(bundle_dir, project, tmp_path / "evaluation", predictions_dir=predictions, device="cpu", plots=False)

    bp_row = run.metrics[run.metrics["target"] == "bp"].iloc[0]
    assert bp_row["n_pixels"] == MASK_PIXELS
    assert bp_row["mae"] == pytest.approx(0.0, abs=1e-7)


def test_evaluate_predicts_scores_by_firezone_and_can_rescore_in_place(bundle_dir: Path, project: Path, tmp_path: Path):
    _write_burnp3_results(project)
    output_dir = tmp_path / "evaluation"

    run = run_evaluate(bundle_dir, project, output_dir, device="cpu", plots=False, by_firezone=True, metrics=["ccc", "mae", "bias"])

    predictions = output_dir / PREDICTIONS_DIRNAME
    assert (predictions / "hex01" / "hex01_bp.tif").is_file()
    assert (predictions / "predict.log").is_file()
    assert json.loads((predictions / RUN_MANIFEST_FILENAME).read_text())["status"] == "success"
    assert run.fire_size_note is not None
    assert list(run.metrics.columns) == ["hexel", "target", "area", "n_pixels", "ccc", "mae", "bias"]
    assert run.metrics["mae"].notna().all()
    assert (run.metrics["n_pixels"] == MASK_PIXELS).all()
    zones = pd.read_csv(output_dir / METRICS_PER_FIREZONE_FILENAME)
    assert sorted(set(zones["firezone"])) == [21, 22]
    assert zones.groupby("target")["n_pixels"].sum().eq(MASK_PIXELS).all()
    log = (output_dir / "evaluate.log").read_text()
    assert "Predicting into" in log
    assert "hex01: evaluated in" in log

    rescored = run_evaluate(bundle_dir, project, output_dir, predictions_dir=predictions, device="cpu", plots=False, overwrite=True)

    assert (predictions / "hex01" / "hex01_bp.tif").is_file()
    assert rescored.warnings == []
    pd.testing.assert_series_equal(
        rescored.metrics.set_index("target")["mae"], run.metrics.set_index("target")["mae"], check_exact=False, rtol=1e-6
    )
    assert not (output_dir / METRICS_PER_FIREZONE_FILENAME).exists()


def test_missing_burnp3_outputs_stop_before_predicting(bundle_dir: Path, project: Path, tmp_path: Path):
    output_dir = tmp_path / "evaluation"

    with pytest.raises(EvaluateError, match="nothing was evaluated") as excinfo:
        run_evaluate(bundle_dir, project, output_dir, device="cpu")

    assert "Missing BurnP3+ burn probability output" in str(excinfo.value)
    assert not (output_dir / PREDICTIONS_DIRNAME).exists()
    assert json.loads((output_dir / EVALUATION_MANIFEST_FILENAME).read_text())["status"] == "failed"


def test_predictions_from_another_model_are_flagged(bundle_dir: Path, bundle: ModelBundle, project: Path, tmp_path: Path):
    grids = _write_burnp3_results(project)
    predictions = _write_predictions(
        tmp_path / "predictions", bundle, grids, manifest={"bundle": {"name": "other-model", "version": "0.1.0", "weights_sha256": "abc"}}
    )

    run = run_evaluate(bundle_dir, project, tmp_path / "evaluation", predictions_dir=predictions, device="cpu", plots=False)

    assert run.warnings == [
        "The predictions were made with other-model v0.1.0, not with this bundle; the metrics describe those predictions."
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"metrics": ["ccc", "accuracy"]}, r"Unknown metric\(s\) \['accuracy'\]"),
        ({"mask_scope": "buffer"}, "The predictions cover the 'actual' area"),
        ({"hex_ids": ["7"]}, r"No predictions for hexel\(s\) \['7'\]"),
    ],
)
def test_invalid_requests_are_rejected(bundle_dir: Path, bundle: ModelBundle, project: Path, tmp_path: Path, kwargs, message):
    predictions = _write_predictions(tmp_path / "predictions", bundle, _write_burnp3_results(project))

    with pytest.raises(EvaluateError, match=message):
        run_evaluate(bundle_dir, project, tmp_path / "evaluation", predictions_dir=predictions, **kwargs)


def test_evaluate_cli_prints_summary_and_plain_errors(bundle_dir: Path, bundle: ModelBundle, project: Path, tmp_path: Path, capsys):
    predictions = _write_predictions(tmp_path / "predictions", bundle, _write_burnp3_results(project))
    args = ["--bundle", str(bundle_dir), "--project", str(project), "--predictions", str(predictions), "--device", "cpu", "--no_plots"]

    assert main([*args, "--output", str(tmp_path / "evaluation")]) == 0
    printed = capsys.readouterr().out
    assert "Evaluated 1 hexel(s) against BurnP3+ (area: actual)" in printed
    assert "ccc" in printed
    assert "Hazard classes: exact accuracy 1.000" in printed

    assert main([*args, "--output", str(tmp_path / "evaluation")]) == 2
    assert "is not empty" in capsys.readouterr().err


def test_evaluate_without_mask_scores_the_whole_raster(bundle_dir: Path, project: Path, tmp_path: Path):
    _write_burnp3_results(project)
    for path in (project / "hex01" / "spatial" / "mask_grids").iterdir():
        path.unlink()

    run = run_evaluate(bundle_dir, project, tmp_path / "evaluation", device="cpu", plots=False, mask_scope="none", metrics=["mae"])

    assert (run.metrics["area"] == "none").all()
    assert (run.metrics["n_pixels"] == HEIGHT * WIDTH).all()
    rescored = run_evaluate(
        bundle_dir,
        project,
        tmp_path / "evaluation",
        predictions_dir=tmp_path / "evaluation" / PREDICTIONS_DIRNAME,
        device="cpu",
        plots=False,
        metrics=["mae"],
        overwrite=True,
    )
    assert (rescored.metrics["area"] == "none").all()

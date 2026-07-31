"""Tests for src/evaluate_hazard.py that avoid real model inference."""

import json
from pathlib import Path

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin

from src.config import Config, HazardEvalConfig, HazardModelEntry
from src.datasets.postprocessing.hexel_reconstruction import StitchedHexel
from src.datasets.targets import get_target_spec
from src.evaluate_hazard import (
    _write_hazard_metric_summaries_from_records,
    load_hazard_config,
    parse_args,
    prepare_model_config_for_hazard,
    raw_ground_truth_denominator,
    read_reference_denominator,
    resolve_hazard_denominator,
    row_normalized_confusion_percentages,
)

BP_CONFIG = Path("configs/bp_spatial_weather.yaml")
MULTI_OUTPUT_CONFIG = Path("configs/multi_output_spatial_weather.yaml")
HAZARD_MODEL_CONFIG = Path("configs/multi_output_spatial_weather.yaml")
HAZARD_EVAL_CONFIG = Path("configs/hazard_eval_spatial_weather.yaml")


def _profile(height=2, width=2):
    return {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": CRS.from_epsg(3978),
        "transform": from_origin(0.0, float(height), 1.0, 1.0),
        "nodata": -9999.0,
    }


def _hexel(target_name, pred_grid, gt_grid, hex_id="01"):
    return StitchedHexel(
        hex_id=hex_id,
        target=get_target_spec(target_name),
        gt_grid=np.asarray(gt_grid, dtype=float),
        pred_grid=np.asarray(pred_grid, dtype=float),
        profile=_profile(),
    )


def _paired_hexels():
    bp = _hexel("bp", pred_grid=[[0.5, 0.0], [0.2, 0.1]], gt_grid=[[0.4, 0.1], [0.3, 0.0]])
    fi = _hexel("fi", pred_grid=[[100.0, 200.0], [50.0, 50.0]], gt_grid=[[80.0, 120.0], [60.0, 40.0]])
    return [(bp, fi)]


def _pair(hex_id, bp_pred, fi_pred, bp_gt=None, fi_gt=None):
    if bp_gt is None:
        bp_gt = bp_pred
    if fi_gt is None:
        fi_gt = fi_pred
    return (
        _hexel("bp", pred_grid=bp_pred, gt_grid=bp_gt, hex_id=hex_id),
        _hexel("fi", pred_grid=fi_pred, gt_grid=fi_gt, hex_id=hex_id),
    )


def _hazard_config(**overrides):
    base = dict(
        root_dir="placeholder/root",
        raw_data_dir="placeholder/raw",
        model=HazardModelEntry(config_path=str(MULTI_OUTPUT_CONFIG)),
    )
    base.update(overrides)
    return HazardEvalConfig(**base)


class TestCliOverrides:
    def _parse(self, monkeypatch, argv):
        monkeypatch.setattr("sys.argv", ["evaluate_hazard", *argv])
        return parse_args()

    def _apply(self, hazard_config, args):
        overrides = {
            key: value
            for key, value in (
                ("mask_scope", args.mask_scope),
                ("save_dir", args.save_dir),
                ("root_dir", args.root_dir),
                ("stitch_mode", args.stitch_mode),
                ("self_normalized_prediction", True if args.self_normalized_prediction else None),
            )
            if value is not None
        }
        return hazard_config.model_copy(update=overrides) if overrides else hazard_config

    def test_defaults_are_none(self, monkeypatch):
        args = self._parse(monkeypatch, ["--config", str(HAZARD_EVAL_CONFIG)])
        assert args.mask_scope is None
        assert args.save_dir is None
        assert args.root_dir is None
        assert args.stitch_mode is None
        assert args.self_normalized_prediction is False

    def test_no_overrides_returns_same_config(self, monkeypatch):
        args = self._parse(monkeypatch, ["--config", str(HAZARD_EVAL_CONFIG)])
        hazard_config = _hazard_config(mask_scope="actual", stitch_mode="mean")
        assert self._apply(hazard_config, args) is hazard_config

    def test_overrides_applied_and_other_fields_preserved(self, monkeypatch):
        args = self._parse(
            monkeypatch,
            [
                "--config",
                str(HAZARD_EVAL_CONFIG),
                "--mask_scope",
                "buffer_only",
                "--save_dir",
                "experiments/hazard_eval/buffer_only",
                "--root_dir",
                "/tmp/buffer-root",
                "--stitch_mode",
                "max",
            ],
        )
        hazard_config = _hazard_config(mask_scope="actual", stitch_mode="mean", save_dir="experiments/original")
        updated = self._apply(hazard_config, args)

        assert updated.mask_scope == "buffer_only"
        assert updated.save_dir == "experiments/hazard_eval/buffer_only"
        assert updated.root_dir == "/tmp/buffer-root"
        assert updated.stitch_mode == "max"
        assert hazard_config.mask_scope == "actual"

    def test_self_normalized_prediction_override(self, monkeypatch):
        args = self._parse(monkeypatch, ["--config", str(HAZARD_EVAL_CONFIG), "--self_normalized_prediction"])
        hazard_config = _hazard_config(self_normalized_prediction=False)
        updated = self._apply(hazard_config, args)

        assert updated.self_normalized_prediction is True

    def test_partial_override_only_touches_given_field(self, monkeypatch):
        args = self._parse(monkeypatch, ["--config", str(HAZARD_EVAL_CONFIG), "--mask_scope", "buffer"])
        hazard_config = _hazard_config(mask_scope="actual", stitch_mode="mean", save_dir="experiments/keep")
        updated = self._apply(hazard_config, args)

        assert updated.mask_scope == "buffer"
        assert updated.save_dir == "experiments/keep"
        assert updated.root_dir == hazard_config.root_dir
        assert updated.stitch_mode == "mean"

    @pytest.mark.parametrize(("flag", "value"), [("--mask_scope", "invalid"), ("--stitch_mode", "median")])
    def test_rejects_invalid_choices(self, monkeypatch, flag, value):
        with pytest.raises(SystemExit):
            self._parse(monkeypatch, ["--config", str(HAZARD_EVAL_CONFIG), flag, value])


class TestLoadHazardConfig:
    def test_loads_real_yaml(self):
        config = load_hazard_config(str(HAZARD_EVAL_CONFIG))
        assert isinstance(config, HazardEvalConfig)
        assert config.model.config_path == str(HAZARD_MODEL_CONFIG)


class TestPrepareModelConfigForHazard:
    def _multi_output_config(self) -> Config:
        import yaml

        with MULTI_OUTPUT_CONFIG.open() as handle:
            return Config(**yaml.safe_load(handle))

    def _bp_config(self) -> Config:
        import yaml

        with BP_CONFIG.open() as handle:
            return Config(**yaml.safe_load(handle))

    def test_overrides_paths_checkpoint_and_logger(self):
        model_config = self._multi_output_config()
        hazard_config = _hazard_config(
            root_dir="/staged/root",
            raw_data_dir="/persistent/raw",
            test_split="custom_test.csv",
            valid_mask_threshold=0.5,
            model=HazardModelEntry(config_path=str(MULTI_OUTPUT_CONFIG), checkpoint_filename="epoch_10.pth"),
        )
        prepared = prepare_model_config_for_hazard(model_config, hazard_config, hazard_config.model)

        assert prepared.data.root_dir == "/staged/root"
        assert prepared.data.raw_data_dir == "/persistent/raw"
        assert prepared.data.test_split == "custom_test.csv"
        assert prepared.data.valid_mask_threshold == 0.5
        assert prepared.evaluation.checkpoint_filename == "epoch_10.pth"
        assert prepared.logger.enabled is False

    def test_rejects_model_missing_required_targets(self):
        model_config = self._bp_config()
        hazard_config = _hazard_config()
        with pytest.raises(ValueError, match="multi-output model predicting"):
            prepare_model_config_for_hazard(model_config, hazard_config, hazard_config.model)


class TestReadReferenceDenominator:
    def test_reads_bare_number(self, tmp_path):
        path = tmp_path / "denom.json"
        path.write_text("123.5")
        assert read_reference_denominator(str(path)) == 123.5

    @pytest.mark.parametrize("key", ["scale_denominator", "denominator"])
    def test_reads_dict_keys(self, tmp_path, key):
        path = tmp_path / "denom.json"
        path.write_text(json.dumps({key: 42.0}))
        assert read_reference_denominator(str(path)) == 42.0

    def test_rejects_missing_key(self, tmp_path):
        path = tmp_path / "denom.json"
        path.write_text(json.dumps({"unrelated": 1.0}))
        with pytest.raises(KeyError, match="scale_denominator"):
            read_reference_denominator(str(path))

    @pytest.mark.parametrize("value", ["0", "-5", "NaN", "Infinity"])
    def test_rejects_non_positive_or_non_finite_bare(self, tmp_path, value):
        path = tmp_path / "denom.json"
        path.write_text(value)
        with pytest.raises(ValueError, match="positive finite"):
            read_reference_denominator(str(path))

    @pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_non_positive_or_non_finite_dict(self, tmp_path, value):
        path = tmp_path / "denom.json"
        path.write_text(json.dumps({"scale_denominator": value}))
        with pytest.raises(ValueError, match="positive finite"):
            read_reference_denominator(str(path))


class TestResolveHazardDenominator:
    def test_explicit_denominator_wins(self):
        hazard_config = _hazard_config(scale_denominator=7.0, scale_denominator_source="eval_ground_truth")
        denom, meta = resolve_hazard_denominator(hazard_config, _paired_hexels())
        assert denom == 7.0
        assert meta["source"] == "explicit"

    def test_eval_ground_truth_max(self):
        # gt raw = bp_gt * min(fi_gt, cap): max is 0.4*80=32, 0.1*120=12, 0.3*60=18, 0.0
        hazard_config = _hazard_config(scale_denominator_source="eval_ground_truth")
        denom, meta = resolve_hazard_denominator(hazard_config, _paired_hexels())
        assert denom == 32.0
        assert meta["source"] == "eval_ground_truth"

    def test_prediction_max(self):
        # pred raw = bp_pred * min(fi_pred, cap): 0.5*100=50, 0.0*200=0, 0.2*50=10, 0.1*50=5
        hazard_config = _hazard_config(scale_denominator_source="prediction")
        denom, _ = resolve_hazard_denominator(hazard_config, _paired_hexels())
        assert denom == 50.0

    def test_eval_ground_truth_skips_empty_or_non_positive_hexels(self):
        hazard_config = _hazard_config(scale_denominator_source="eval_ground_truth")
        pairs = [
            _pair("01", bp_pred=[[0.0]], fi_pred=[[0.0]]),
            _pair("02", bp_pred=[[np.nan]], fi_pred=[[1.0]]),
            _pair("03", bp_pred=[[0.5]], fi_pred=[[20.0]]),
        ]

        denom, _ = resolve_hazard_denominator(hazard_config, pairs)

        assert denom == 10.0

    def test_prediction_denominator_raises_only_when_no_positive_hazard_exists(self):
        hazard_config = _hazard_config(scale_denominator_source="prediction")
        pairs = [
            _pair("01", bp_pred=[[0.0]], fi_pred=[[100.0]]),
            _pair("02", bp_pred=[[np.nan]], fi_pred=[[100.0]]),
        ]

        with pytest.raises(ValueError, match="maximum finite hazard must be > 0|no finite hazard"):
            resolve_hazard_denominator(hazard_config, pairs)

    def test_all_raw_ground_truth_writes_json(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "src.evaluate_hazard.raw_ground_truth_denominator",
            lambda raw_data_dir, fi_cap, *args, **kwargs: 99.0,
        )
        hazard_config = _hazard_config(scale_denominator_source="all_raw_ground_truth")
        denom, meta = resolve_hazard_denominator(hazard_config, _paired_hexels(), save_dir=str(tmp_path))
        assert denom == 99.0
        written = json.loads((tmp_path / "hazard_scale_denominator.json").read_text())
        assert written["scale_denominator"] == 99.0
        assert written["scale_denominator_source"] == "all_raw_ground_truth"


class TestRawGroundTruthDenominator:
    def test_max_over_raw_rasters(self, monkeypatch):
        # Two hexes; hazard = bp * min(fi, cap). Expected max is hex "02": 0.5 * 200 = 100.
        rasters = {
            ("01", "bp"): np.array([[0.1, 0.2]]),
            ("01", "fi"): np.array([[10.0, 20.0]]),
            ("02", "bp"): np.array([[0.5, 0.0]]),
            ("02", "fi"): np.array([[200.0, 5.0]]),
        }

        class FakePaths:
            def __init__(self, hex_id, root_dir):
                self.hex_id = hex_id

            def output_burn_prob(self):
                return f"{self.hex_id}:bp"

            def output_fire_intensity(self):
                return f"{self.hex_id}:fi"

        def fake_load_raster(path):
            hex_id, kind = path.split(":")
            return rasters[(hex_id, kind)]

        monkeypatch.setattr("src.evaluate_hazard.find_hex_ids", lambda root_dir: ["01", "02"])
        monkeypatch.setattr("src.evaluate_hazard.Paths", FakePaths)
        monkeypatch.setattr("src.evaluate_hazard.load_raster", fake_load_raster)

        assert raw_ground_truth_denominator("/raw", fi_cap=10000.0) == 100.0

    def test_respects_fi_cap(self, monkeypatch):
        rasters = {("01", "bp"): np.array([[1.0]]), ("01", "fi"): np.array([[50000.0]])}

        class FakePaths:
            def __init__(self, hex_id, root_dir):
                self.hex_id = hex_id

            def output_burn_prob(self):
                return f"{self.hex_id}:bp"

            def output_fire_intensity(self):
                return f"{self.hex_id}:fi"

        monkeypatch.setattr("src.evaluate_hazard.find_hex_ids", lambda root_dir: ["01"])
        monkeypatch.setattr("src.evaluate_hazard.Paths", FakePaths)
        monkeypatch.setattr("src.evaluate_hazard.load_raster", lambda path: rasters[tuple(path.split(":"))])

        assert raw_ground_truth_denominator("/raw", fi_cap=10000.0) == 10000.0

    def test_restricts_to_allowed_hex_ids(self, monkeypatch):
        # Only hex "01" is allowed, so hex "02" (larger hazard) must be excluded.
        rasters = {
            ("01", "bp"): np.array([[0.1, 0.2]]),
            ("01", "fi"): np.array([[10.0, 20.0]]),
            ("02", "bp"): np.array([[0.5, 0.0]]),
            ("02", "fi"): np.array([[200.0, 5.0]]),
        }

        class FakePaths:
            def __init__(self, hex_id, root_dir):
                self.hex_id = hex_id

            def output_burn_prob(self):
                return f"{self.hex_id}:bp"

            def output_fire_intensity(self):
                return f"{self.hex_id}:fi"

        def fake_load_raster(path):
            hex_id, kind = path.split(":")
            return rasters[(hex_id, kind)]

        monkeypatch.setattr("src.evaluate_hazard.find_hex_ids", lambda root_dir: ["01", "02"])
        monkeypatch.setattr("src.evaluate_hazard.Paths", FakePaths)
        monkeypatch.setattr("src.evaluate_hazard.load_raster", fake_load_raster)

        # Max over hex "01" only: 0.2 * 20 = 4.0.
        assert raw_ground_truth_denominator("/raw", fi_cap=10000.0, allowed_hex_ids=[1]) == pytest.approx(4.0)


class TestWriteHazardMetricSummaries:
    def test_row_normalizes_confusion_matrix_percentages(self):
        confusion = np.array([[3, 1], [0, 0]])
        percentages = row_normalized_confusion_percentages(confusion)

        np.testing.assert_allclose(percentages[0], [75.0, 25.0])
        assert np.isnan(percentages[1]).all()

    def test_writes_per_hex_csv_and_aggregate_json(self, tmp_path):
        metric_records = [
            {"hex_id": "01", "exact_accuracy": 0.8, "per_class_iou_1": 0.5, "per_class_iou_2": np.nan},
            {"hex_id": "02", "exact_accuracy": 0.6, "per_class_iou_1": 0.7, "per_class_iou_2": 0.3},
        ]
        csv_path, json_path, aggregate = _write_hazard_metric_summaries_from_records(
            metric_records,
            [],
            str(tmp_path),
            denominator=50.0,
            denominator_metadata={"source": "prediction", "value": 50.0},
        )

        import pandas as pd

        per_hex = pd.read_csv(csv_path, dtype={"hex_id": str})
        assert list(per_hex["hex_id"]) == ["01", "02"]
        assert set(per_hex.columns) >= {"hex_id", "exact_accuracy", "per_class_iou_1", "per_class_iou_2"}

        summary = json.loads(Path(json_path).read_text())
        assert summary["num_hexels"] == 2
        assert summary["denominator"] == 50.0
        assert summary["denominator_metadata"]["source"] == "prediction"
        # finite mean ignores the NaN per_class_iou_2 value from hex 01
        assert aggregate["exact_accuracy"] == pytest.approx(0.7)
        assert aggregate["per_class_iou_2"] == pytest.approx(0.3)
        assert summary["metrics"]["per_class_iou_1"] == pytest.approx(0.6)

    def test_writes_confusion_matrix_outputs(self, tmp_path):
        _, json_path, _ = _write_hazard_metric_summaries_from_records(
            [{"hex_id": "01", "exact_accuracy": 0.8}, {"hex_id": "02", "exact_accuracy": 0.6}],
            [np.array([[2, 1], [0, 3]]), np.array([[1, 0], [2, 4]])],
            str(tmp_path),
            denominator=50.0,
            denominator_metadata={"source": "all_raw_ground_truth", "value": 50.0},
            prediction_denominator=25.0,
            prediction_denominator_metadata={"source": "prediction", "value": 25.0},
        )

        summary = json.loads(Path(json_path).read_text())
        assert summary["prediction_denominator"] == 25.0
        assert summary["confusion_matrix"] == [[3, 1], [2, 7]]
        assert Path(summary["confusion_matrix_csv"]).is_file()
        assert Path(summary["confusion_matrix_plot"]).is_file()

    def test_all_nan_aggregate_becomes_null(self, tmp_path):
        _, json_path, aggregate = _write_hazard_metric_summaries_from_records(
            [{"hex_id": "01", "exact_accuracy": np.nan}, {"hex_id": "02", "exact_accuracy": np.nan}],
            [],
            str(tmp_path),
            denominator=1.0,
            denominator_metadata={"source": "prediction"},
        )

        assert np.isnan(aggregate["exact_accuracy"])
        raw = Path(json_path).read_text()
        assert "NaN" not in raw
        summary = json.loads(raw)
        assert summary["metrics"]["exact_accuracy"] is None

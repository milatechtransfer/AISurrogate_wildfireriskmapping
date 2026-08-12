import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from src.datasets.postprocessing.hazard_pipeline import (
    compute_hazard_hexel,
    pair_stitched_hexels,
    save_hazard_hexel_artifacts,
)
from src.datasets.postprocessing.hexel_reconstruction import StitchedHexel
from src.datasets.targets import get_target_spec


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
    bp = _hexel(
        "bp",
        pred_grid=[[0.5, 0.0], [0.2, np.nan]],
        gt_grid=[[0.4, 0.1], [0.3, 0.0]],
    )
    fi = _hexel(
        "fi",
        pred_grid=[[100.0, 200.0], [np.nan, 50.0]],
        gt_grid=[[80.0, 120.0], [60.0, 40.0]],
    )
    return bp, fi


class TestPairStitchedHexels:
    def test_pairs_matching_sequence(self):
        hexels = [
            _hexel("bp", [[0.1]], [[0.1]], hex_id="01"),
            _hexel("fi", [[1.0]], [[1.0]], hex_id="01"),
            _hexel("bp", [[0.2]], [[0.2]], hex_id="02"),
            _hexel("fi", [[2.0]], [[2.0]], hex_id="02"),
        ]
        pairs = list(pair_stitched_hexels(hexels))
        assert [p[0].hex_id for p in pairs] == ["01", "02"]
        assert [p[0].target.name for p in pairs] == ["bp", "bp"]
        assert [p[1].target.name for p in pairs] == ["fi", "fi"]

    def test_ignores_extra_targets(self):
        hexels = [
            _hexel("bp", [[0.1]], [[0.1]], hex_id="01"),
            _hexel("fi", [[1.0]], [[1.0]], hex_id="01"),
            _hexel("ros", [[3.0]], [[3.0]], hex_id="01"),
        ]
        pairs = list(pair_stitched_hexels(hexels))
        assert len(pairs) == 1
        assert pairs[0][0].target.name == "bp"
        assert pairs[0][1].target.name == "fi"

    def test_missing_target_raises(self):
        hexels = [_hexel("bp", [[0.1]], [[0.1]], hex_id="01")]
        with pytest.raises(ValueError, match="missing required target"):
            list(pair_stitched_hexels(hexels))

    def test_duplicate_target_raises(self):
        hexels = [
            _hexel("bp", [[0.1]], [[0.1]], hex_id="01"),
            _hexel("bp", [[0.2]], [[0.2]], hex_id="01"),
            _hexel("fi", [[1.0]], [[1.0]], hex_id="01"),
        ]
        with pytest.raises(ValueError, match="Duplicate"):
            list(pair_stitched_hexels(hexels))


class TestComputeHazardHexel:
    def test_expected_raw_scaled_binned_and_metrics(self):
        bp, fi = _paired_hexels()
        result = compute_hazard_hexel(
            bp,
            fi,
            denominator=50.0,
            scale_to=100.0,
            bin_thresholds=[10.0, 50.0],
        )
        # raw = bp * fi; NaNs propagate
        np.testing.assert_array_equal(result.pred_raw_hazard, np.array([[50.0, 0.0], [np.nan, np.nan]]))
        np.testing.assert_array_equal(result.gt_raw_hazard, np.array([[32.0, 12.0], [18.0, 0.0]]))
        # shared denominator/scale factor of 2 applied to both pred and gt
        np.testing.assert_array_equal(result.pred_scaled_hazard, np.array([[100.0, 0.0], [np.nan, np.nan]]))
        np.testing.assert_array_equal(result.gt_scaled_hazard, np.array([[64.0, 24.0], [36.0, 0.0]]))
        np.testing.assert_array_equal(result.pred_binned_hazard, np.array([[3, 1], [0, 0]]))
        np.testing.assert_array_equal(result.gt_binned_hazard, np.array([[3, 2], [2, 1]]))
        # only the two finite pred pixels are valid: (3,3) exact, (1,2) within-1
        assert result.metrics["exact_accuracy"] == 0.5
        assert result.metrics["within_1_accuracy"] == 1.0
        assert result.metrics["mean_absolute_class_error"] == 0.5

    def test_prediction_denominator_only_scales_prediction(self):
        bp, fi = _paired_hexels()
        result = compute_hazard_hexel(
            bp,
            fi,
            denominator=50.0,
            pred_denominator=100.0,
            scale_to=100.0,
            bin_thresholds=[10.0, 50.0],
        )

        np.testing.assert_array_equal(result.pred_scaled_hazard, np.array([[50.0, 0.0], [np.nan, np.nan]]))
        np.testing.assert_array_equal(result.gt_scaled_hazard, np.array([[64.0, 24.0], [36.0, 0.0]]))

    def test_hex_id_mismatch_raises(self):
        bp = _hexel("bp", [[0.1]], [[0.1]], hex_id="01")
        fi = _hexel("fi", [[1.0]], [[1.0]], hex_id="02")
        with pytest.raises(ValueError, match="hex_id mismatch"):
            compute_hazard_hexel(bp, fi, denominator=1.0)

    def test_wrong_targets_raise(self):
        bp = _hexel("fi", [[0.1]], [[0.1]])
        fi = _hexel("fi", [[1.0]], [[1.0]])
        with pytest.raises(ValueError, match="'bp' target"):
            compute_hazard_hexel(bp, fi, denominator=1.0)

    def test_shape_mismatch_raises(self):
        bp = _hexel("bp", [[0.1, 0.2]], [[0.1, 0.2]])
        fi = _hexel("fi", [[1.0]], [[1.0]])
        with pytest.raises(ValueError, match="share a shape"):
            compute_hazard_hexel(bp, fi, denominator=1.0)

    def test_crs_mismatch_raises(self):
        bp, fi = _paired_hexels()
        fi.profile["crs"] = CRS.from_epsg(4326)
        with pytest.raises(ValueError, match="share crs"):
            compute_hazard_hexel(bp, fi, denominator=1.0)


class TestSaveHazardHexelArtifacts:
    def test_writes_geotiffs_with_nodata(self, tmp_path):
        bp, fi = _paired_hexels()
        result = compute_hazard_hexel(bp, fi, denominator=50.0, scale_to=100.0, bin_thresholds=[10.0, 50.0])
        written = save_hazard_hexel_artifacts(result, str(tmp_path), save_plots=False)
        assert len(written) == 6
        for path in written:
            assert path.endswith(".tif")
            with rasterio.open(path) as src:
                assert src.count == 1
        pred_raw_path = next(p for p in written if "predicted_raw" in p)
        with rasterio.open(pred_raw_path) as src:
            data = src.read(1)
            assert src.nodata == -9999.0
            assert data[1, 0] == -9999.0
            assert data[0, 0] == 50.0

    def test_binned_geotiff_int_dtype_and_nodata(self, tmp_path):
        bp, fi = _paired_hexels()
        result = compute_hazard_hexel(bp, fi, denominator=50.0, scale_to=100.0, bin_thresholds=[10.0, 50.0])
        written = save_hazard_hexel_artifacts(result, str(tmp_path), save_plots=False)
        pred_binned_path = next(p for p in written if "predicted_binned" in p)
        with rasterio.open(pred_binned_path) as src:
            data = src.read(1)
            assert src.dtypes[0] == "int32"
            assert src.nodata == 0
            # invalid (non-finite) cells collapse to the invalid class 0
            assert data[1, 0] == 0
            assert data[1, 1] == 0
            assert data[0, 0] == 3

    def test_save_plots_invokes_visualizer(self, tmp_path, monkeypatch):
        bp, fi = _paired_hexels()
        result = compute_hazard_hexel(bp, fi, denominator=50.0, scale_to=100.0, bin_thresholds=[10.0, 50.0])
        calls = []
        monkeypatch.setattr(
            "src.datasets.postprocessing.hazard_pipeline.visualize_target_grids",
            lambda **kwargs: calls.append(kwargs),
        )
        save_hazard_hexel_artifacts(result, str(tmp_path), save_plots=True)
        suffixes = {c["filename_suffix"] for c in calls}
        assert suffixes == {"_raw", "_scaled"}
        assert all(c["target_name"] == "hazard" for c in calls)

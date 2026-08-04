import numpy as np
import pytest

from src.config import ModelConfig
from src.mean_baseline import MeanBaselineRegressor
from src.train_tabular_baseline import build_baseline_model, denormalize_target, evaluate_region_level


# ---------- MeanBaselineRegressor ----------
def test_mean_baseline_regressor_predicts_training_mean():
    reg = MeanBaselineRegressor()
    X = np.zeros((4, 3), dtype=np.float32)
    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    reg.fit(X, y)

    preds = reg.predict(np.zeros((10, 3), dtype=np.float32))

    assert preds.shape == (10,)
    assert np.all(preds == pytest.approx(2.5))


def test_mean_baseline_regressor_ignores_input_features():
    reg = MeanBaselineRegressor()
    X = np.random.rand(5, 4).astype(np.float32)
    y = np.array([10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)
    reg.fit(X, y)

    preds_a = reg.predict(np.zeros((3, 4), dtype=np.float32))
    preds_b = reg.predict(np.ones((3, 4), dtype=np.float32) * 999.0)

    np.testing.assert_allclose(preds_a, preds_b)
    assert np.all(preds_a == pytest.approx(10.0))


def test_mean_baseline_regressor_rejects_empty_target():
    reg = MeanBaselineRegressor()
    with pytest.raises(ValueError, match="empty target array"):
        reg.fit(np.zeros((0, 3)), np.zeros(0))


def test_mean_baseline_regressor_predict_before_fit_raises():
    reg = MeanBaselineRegressor()
    with pytest.raises(RuntimeError, match="before fit"):
        reg.predict(np.zeros((2, 3)))


# ---------- build_baseline_model dispatch ----------
def test_build_baseline_model_xgboost():
    from xgboost import XGBRegressor

    config = ModelConfig(architecture="xgboost", num_classes=1, params={"n_estimators": 5, "max_depth": 2})
    model = build_baseline_model(config)

    assert isinstance(model, XGBRegressor)


def test_build_baseline_model_mean_baseline():
    config = ModelConfig(architecture="mean_baseline", num_classes=1, params=None)
    model = build_baseline_model(config)

    assert isinstance(model, MeanBaselineRegressor)


def test_build_baseline_model_unknown_architecture_raises():
    config = ModelConfig(architecture="not_a_real_model", num_classes=1, params=None)
    with pytest.raises(ValueError, match="Unknown baseline architecture"):
        build_baseline_model(config)


# ---------- denormalize_target ----------
def test_denormalize_target_log_standard_requires_stats():
    y_norm = np.array([0.0, 1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="target_log_mean and target_log_std are required"):
        denormalize_target(y_norm, "log_standard", target_min=0.0, target_max=1.0, target_log_mean=None, target_log_std=None)


def test_denormalize_target_log_standard_computes_expm1():
    y_norm = np.array([0.0], dtype=np.float32)
    out = denormalize_target(y_norm, "log_standard", target_min=0.0, target_max=1.0, target_log_mean=0.0, target_log_std=1.0)

    assert out[0] == pytest.approx(0.0, abs=1e-6)


def test_denormalize_target_min_max():
    y_norm = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    out = denormalize_target(y_norm, "min_max", target_min=10.0, target_max=20.0, target_log_mean=None, target_log_std=None)

    np.testing.assert_allclose(out, [10.0, 15.0, 20.0])


# ---------- evaluate_region_level empty-metrics guard ----------
def test_evaluate_region_level_raises_on_empty_metrics():
    with pytest.raises(RuntimeError, match="produced no per-hexel metrics"):
        evaluate_region_level(
            model=None,
            loader=[],  # empty loader -> hex_data stays empty -> per_hexel_metrics stays empty
            out_norm="min_max",
            target_min=0.0,
            target_max=1.0,
            target_log_mean=None,
            target_log_std=None,
            valid_mask_threshold=0.01,
            config=None,
            metric_functions={},
            target_name="bp",
            device="cpu",
            max_batches=0,
        )

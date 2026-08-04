"""
Mean-training-value baseline: predicts the scalar mean of the training-pixel
targets (in normalized space) everywhere, ignoring all spatial and fuel-curve
inputs. Zero-spatial-skill floor for comparison against the XGBoost spatial-only
baseline and the U-Net surrogate.
"""

import numpy as np


class MeanBaselineRegressor:
    """Minimal sklearn-compatible constant regressor. Ignores X entirely."""

    def __init__(self) -> None:
        self.mean_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MeanBaselineRegressor":
        if y.size == 0:
            raise ValueError("MeanBaselineRegressor.fit received an empty target array.")
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("MeanBaselineRegressor.predict called before fit().")
        return np.full(X.shape[0], self.mean_, dtype=np.float32)

"""
Pixel-based metrics (MAE, MSE, RMSE).
"""

import numpy as np

from pybox.core.interfaces import MetricStrategy
from .utils import normalize_diff


class MAEMetric(MetricStrategy):
    """Mean Absolute Error metric strategy."""

    def calculate(self, original: np.ndarray, adversarial: np.ndarray) -> np.ndarray:
        diff = np.abs(original - adversarial)
        return normalize_diff(diff)


class MSEMetric(MetricStrategy):
    """Mean Squared Error metric strategy."""

    def calculate(self, original: np.ndarray, adversarial: np.ndarray) -> np.ndarray:
        diff = np.square(original - adversarial)
        return normalize_diff(diff)


class RMSEMetric(MetricStrategy):
    """Root Mean Square Error metric strategy."""

    def calculate(self, original: np.ndarray, adversarial: np.ndarray) -> np.ndarray:
        mse = np.square(original - adversarial)
        diff = np.sqrt(mse)
        return normalize_diff(diff)

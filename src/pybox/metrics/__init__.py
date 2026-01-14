from pybox.core.interfaces import MetricStrategy
from .pixel import MAEMetric, MSEMetric, RMSEMetric
from .perceptual import CosineMetric, GMDMetric

from typing import Literal, Dict


# Literal import for type hinting of metric strings
METRIC_LITERALS = Literal["MAE", "MSE", "RMSE", "COSINE", "GMD"]

# Factory map for convenience
METRIC_MAP: Dict[str, type[MetricStrategy]] = {
    "MAE": MAEMetric,
    "MSE": MSEMetric,
    "RMSE": RMSEMetric,
    "COSINE": CosineMetric,
    "GMD": GMDMetric,
}

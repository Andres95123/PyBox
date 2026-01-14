from .pixel import MAEMetric, MSEMetric, RMSEMetric
from .perceptual import CosineMetric, GMDMetric

# Factory map for convenience
METRIC_MAP = {
    "MAE": MAEMetric,
    "MSE": MSEMetric,
    "RMSE": RMSEMetric,
    "COSINE": CosineMetric,
    "GMD": GMDMetric,
}

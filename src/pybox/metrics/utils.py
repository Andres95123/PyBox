"""
Utility functions for metrics.
"""

import numpy as np


def normalize_diff(diff_img: np.ndarray) -> np.ndarray:
    """
    Normalizes a difference image to [0, 255] uint8.
    Scales the maximum difference to 255 for visualization.
    """
    maxv = np.max(diff_img)
    # Use a threshold to avoid amplifying floating point noise for identical images
    if maxv > 1e-7:
        diff_img = (diff_img / maxv) * 255
    else:
        diff_img = np.zeros_like(diff_img)
    return diff_img.astype(np.uint8)

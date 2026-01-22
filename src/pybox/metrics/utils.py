"""
Utility functions for metrics.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def normalize_diff(diff_img: np.ndarray) -> np.ndarray:
    """
    Normalizes a difference image to [0, 255] uint8.
    Scales the maximum difference to 255 for visualization.
    """
    # Flatten the image to (num_pixels,) for proper scaling
    original_shape = diff_img.shape
    flat_diff = diff_img.reshape(-1, 1)

    # Handle case where all values are the same (min == max)
    min_val = flat_diff.min()
    max_val = flat_diff.max()

    if min_val == max_val:
        # If all values are equal, check if they are non-zero
        # Non-zero constant values should map to 255
        if min_val > 0:
            return np.full(original_shape, 255, dtype=np.uint8)
        else:
            return np.zeros(original_shape, dtype=np.uint8)

    # Apply MinMaxScaler to scale to [0, 255]
    scaler = MinMaxScaler(feature_range=(0, 255))
    scaled_diff = scaler.fit_transform(flat_diff)

    # Reshape back to original shape and convert to uint8
    return scaled_diff.reshape(original_shape).astype(np.uint8)

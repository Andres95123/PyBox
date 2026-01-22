"""
Perceptual metrics (Cosine, GMD).
"""

import numpy as np

# Import from sklearn the cosine and gradient functions
from sklearn.metrics.pairwise import cosine_similarity
from scipy.ndimage import gaussian_gradient_magnitude

from pybox.core.interfaces import MetricStrategy
from .utils import normalize_diff


class CosineMetric(MetricStrategy):
    """Cosine Similarity metric strategy."""

    def __call__(self, original: np.ndarray, adversarial: np.ndarray) -> np.ndarray:
        # Reshape images to (num_pixels, num_channels) for cosine_similarity computation
        orig_pixels = original.reshape(-1, original.shape[-1]).astype(float)
        adv_pixels = adversarial.reshape(-1, adversarial.shape[-1]).astype(float)

        # Compute cosine similarity between corresponding pixels
        sim_matrix = cosine_similarity(orig_pixels, adv_pixels)
        sim = np.diag(sim_matrix)
        diff = 1 - sim

        # Reshape back to image dimensions and expand to 3 channels
        diff_img = diff.reshape(original.shape[:-1])
        dif_img = np.stack([diff_img] * 3, axis=-1)
        return normalize_diff(dif_img)


class GMDMetric(MetricStrategy):
    """Gradient Magnitude Difference metric strategy using Gaussian Gradient Magnitude."""

    def __call__(self, original: np.ndarray, adversarial: np.ndarray) -> np.ndarray:
        # Compute Gaussian gradient magnitude for each channel
        grad1 = np.zeros_like(original, dtype=float)
        grad2 = np.zeros_like(adversarial, dtype=float)

        for c in range(original.shape[2]):
            grad1[:, :, c] = gaussian_gradient_magnitude(
                original[:, :, c].astype(float), sigma=1.0
            )
            grad2[:, :, c] = gaussian_gradient_magnitude(
                adversarial[:, :, c].astype(float), sigma=1.0
            )

        # Compute absolute difference between gradient magnitudes
        diff = np.abs(grad1 - grad2)
        return normalize_diff(diff)

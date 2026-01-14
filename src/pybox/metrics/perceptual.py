"""
Perceptual metrics (Cosine, GMD).
"""

import numpy as np

from pybox.core.interfaces import MetricStrategy
from .utils import normalize_diff


class CosineMetric(MetricStrategy):
    """Cosine Similarity metric strategy."""

    def calculate(self, original: np.ndarray, adversarial: np.ndarray) -> np.ndarray:
        def cosine_sim(a, b):
            dot = np.sum(a * b, axis=-1)
            norm_a = np.linalg.norm(a, axis=-1)
            norm_b = np.linalg.norm(b, axis=-1)
            return dot / (norm_a * norm_b + 1e-10)

        sim = cosine_sim(original.astype(float), adversarial.astype(float))
        diff = 1 - sim
        # Expand to 3 channels for consistency (as per original implementation)
        dif_img = np.stack([diff] * 3, axis=-1)
        return normalize_diff(dif_img)


class GMDMetric(MetricStrategy):
    """Gradient Magnitude Difference metric strategy."""

    def calculate(self, original: np.ndarray, adversarial: np.ndarray) -> np.ndarray:
        def compute_grad(img):
            grad = np.zeros_like(img, dtype=float)
            for c in range(img.shape[2]):
                ch = img[:, :, c].astype(float)
                grad_y, grad_x = np.gradient(ch)
                grad[:, :, c] = np.sqrt(grad_x**2 + grad_y**2)
            return grad

        g1 = compute_grad(original)
        g2 = compute_grad(adversarial)
        diff = np.abs(g1 - g2)
        return normalize_diff(diff)

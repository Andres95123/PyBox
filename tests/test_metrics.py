import numpy as np
import pytest
from pybox.metrics import MAEMetric, MSEMetric, RMSEMetric, CosineMetric, GMDMetric


@pytest.fixture
def sample_images():
    img1 = np.zeros((10, 10, 3), dtype=np.uint8)
    img2 = np.ones((10, 10, 3), dtype=np.uint8) * 255  # All white
    return img1, img2


def test_mae_metric(sample_images):
    img1, img2 = sample_images
    metric = MAEMetric()
    diff = metric.calculate(img1, img2)
    assert diff.shape == img1.shape
    assert diff.dtype == np.uint8
    assert diff.min() > 250  # Should be 255 normalized


def test_shape_consistency():
    metric = MAEMetric()
    img1 = np.random.rand(28, 28, 3)
    img2 = np.random.rand(28, 28, 3)
    diff = metric.calculate(img1, img2)
    assert diff.shape == (28, 28, 3)


def test_zero_diff():
    metric = MAEMetric()
    img1 = np.random.rand(28, 28, 3)
    diff = metric.calculate(img1, img1)
    assert diff.sum() == 0


def test_cosine_metric():
    metric = CosineMetric()
    img1 = np.ones((10, 10, 3))
    img2 = np.ones((10, 10, 3))
    # Identical images -> cosine sim = 1 -> diff = 0
    diff = metric.calculate(img1, img2)
    assert diff.sum() == 0

    img3 = -1 * img1  # Opposite
    # Cosine sim = -1 (if allowed) or 0 depending on implementation details
    # Implementation: (u . v) / |u||v|. 1.-1 / 1.1 = -1. Diff = 1 - (-1) = 2.
    # Scaled to 255.
    diff_opp = metric.calculate(img1, img3)
    assert diff_opp.max() == 255

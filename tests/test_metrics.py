import numpy as np
import pytest
from pybox.metrics import MAEMetric, MSEMetric, RMSEMetric, CosineMetric, GMDMetric
from pybox.metrics.utils import normalize_diff


@pytest.fixture
def sample_images():
    img1 = np.zeros((10, 10, 3), dtype=np.uint8)
    img2 = np.ones((10, 10, 3), dtype=np.uint8) * 255  # All white
    return img1, img2


def test_mae_metric(sample_images):
    img1, img2 = sample_images
    metric = MAEMetric()
    diff = metric(img1, img2)
    assert diff.shape == img1.shape
    assert diff.dtype == np.uint8
    assert diff.min() > 250  # Should be 255 normalized

    # Test usage by hand
    diff_manual = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    diff_manual = normalize_diff(diff_manual)
    assert np.array_equal(diff, diff_manual)


def test_mse_metric(sample_images):
    img1, img2 = sample_images
    metric = MSEMetric()
    diff = metric(img1, img2)
    assert diff.shape == img1.shape
    assert diff.dtype == np.uint8
    assert diff.min() > 250  # Should be 255 normalized

    # Test usage by hand
    diff_manual = np.square(img1.astype(np.float32) - img2.astype(np.float32))
    diff_manual = normalize_diff(diff_manual)
    assert np.array_equal(diff, diff_manual)


def test_rmse_metric(sample_images):
    img1, img2 = sample_images
    metric = RMSEMetric()
    diff = metric(img1, img2)
    assert diff.shape == img1.shape
    assert diff.dtype == np.uint8
    assert diff.min() > 250  # Should be 255 normalized

    # Test usage by hand
    diff_manual = np.square(img1.astype(np.float32) - img2.astype(np.float32))
    diff_manual = np.sqrt(diff_manual)
    diff_manual = normalize_diff(diff_manual)
    assert np.array_equal(diff, diff_manual)


def test_normalize_diff():
    diff_img = np.array([[0.0, 0.5], [1.0, 2.0]], dtype=np.float32)
    normalized = normalize_diff(diff_img)
    assert normalized.dtype == np.uint8
    assert normalized.max() == 255
    assert normalized.min() == 0


def test_shape_consistency():
    metric = MAEMetric()
    img1 = np.random.rand(28, 28, 3)
    img2 = np.random.rand(28, 28, 3)
    diff = metric(img1, img2)
    assert diff.shape == (28, 28, 3)


def test_zero_diff():
    metric = MAEMetric()
    img1 = np.random.rand(28, 28, 3)
    diff = metric(img1, img1)
    assert diff.sum() == 0


def test_cosine_metric():
    metric = CosineMetric()
    img1 = np.ones((10, 10, 3))
    img2 = np.ones((10, 10, 3))
    # Identical images -> cosine sim = 1 -> diff = 0
    diff = metric(img1, img2)
    assert diff.sum() == 0

    img3 = -1 * img1  # Opposite
    # Cosine sim = -1 (if allowed) or 0 depending on implementation details
    # Implementation: (u . v) / |u||v|. 1.-1 / 1.1 = -1. Diff = 1 - (-1) = 2.
    # Scaled to 255.
    diff_opp = metric(img1, img3)
    assert diff_opp.max() == 255

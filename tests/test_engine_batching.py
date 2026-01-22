import numpy as np
import pytest
from pybox.engine import AdversarialEngine
from pybox.metrics import MAEMetric
from pybox.attacks.art_adapter import ArtAttackAdapter


class MockClassifier:
    def predict(self, x):
        # Predict class 0 for all
        return np.array([[1.0, 0.0]] * len(x))


class MockArtAttack:
    def __init__(self, classifier=None):
        self.last_x_shape = None
        self.last_y = None

    def generate(self, x, y=None):
        self.last_x_shape = x.shape
        self.last_y = y
        # Return modified image to ensure flow is correct
        return x + 0.1


def test_batch_processing_size():
    clf = MockClassifier()
    art_atk = MockArtAttack()
    adapter = ArtAttackAdapter(art_atk, name="MockAttack")
    metric = MAEMetric()

    engine = AdversarialEngine(clf, [adapter], metric)

    # 4 images
    images = np.zeros((4, 28, 28, 3))
    labels = np.zeros(4)

    # Run with batch_size=2
    results = engine.run(images, ground_truth=labels, batch_size=2, verbose=False)

    assert len(results) == 4
    # Check that process_batch was used correctly by verifying result structure
    assert results[0].original_image.shape == (28, 28, 3)


def test_attack_receives_batch():
    clf = MockClassifier()
    art_atk = MockArtAttack()
    adapter = ArtAttackAdapter(art_atk, name="MockAttack")
    metric = MAEMetric()

    engine = AdversarialEngine(clf, [adapter], metric)

    images = np.zeros((4, 28, 28, 3))

    # Run with batch_size=2
    engine.run(images, batch_size=2, verbose=False)

    # The last call should have been with batch size 2 (4 images / 2 = 2 batches, last is full)
    assert art_atk.last_x_shape == (2, 28, 28, 3)


def test_attack_receives_targets_batch():
    clf = MockClassifier()
    art_atk = MockArtAttack()
    adapter = ArtAttackAdapter(art_atk, name="MockAttack")
    metric = MAEMetric()

    engine = AdversarialEngine(clf, [adapter], metric)

    images = np.zeros((4, 28, 28, 3))
    targets = np.array([1, 1, 1, 1])

    engine.run(images, target_labels=targets, batch_size=2, verbose=False)

    assert art_atk.last_y is not None
    assert len(art_atk.last_y) == 2
    assert np.all(art_atk.last_y == 1)


def test_batch_size_larger_than_data():
    clf = MockClassifier()
    art_atk = MockArtAttack()
    adapter = ArtAttackAdapter(art_atk, name="MockAttack")
    metric = MAEMetric()

    engine = AdversarialEngine(clf, [adapter], metric)

    images = np.zeros((3, 28, 28, 3))

    # Run with batch_size=10
    results = engine.run(images, batch_size=10, verbose=False)

    assert len(results) == 3
    assert art_atk.last_x_shape == (3, 28, 28, 3)


def test_batch_remainder():
    clf = MockClassifier()
    art_atk = MockArtAttack()
    adapter = ArtAttackAdapter(art_atk, name="MockAttack")
    metric = MAEMetric()

    engine = AdversarialEngine(clf, [adapter], metric)

    images = np.zeros((5, 28, 28, 3))

    # Run with batch_size=2. Batches: 2, 2, 1.
    engine.run(images, batch_size=2, verbose=False)

    # Last call should be remaining 1
    assert art_atk.last_x_shape == (1, 28, 28, 3)

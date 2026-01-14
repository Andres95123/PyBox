import numpy as np
import pytest
from pybox.engine import AdversarialEngine
from pybox.metrics import MAEMetric
from pybox.attacks.art_adapter import ArtAttackAdapter
from pybox.core.models import ExperimentResult


class MockClassifier:
    def predict(self, x):
        return np.array([[1.0, 0.0]] * len(x))


class MockArtAttack:
    def __init__(self, classifier):
        pass

    def generate(self, x, y=None):
        return x  # No change


def test_engine_basic_flow():
    clf = MockClassifier()
    art_atk = MockArtAttack(clf)
    adapter = ArtAttackAdapter(art_atk, name="MockAttack")
    metric = MAEMetric()

    engine = AdversarialEngine(clf, [adapter], metric)

    images = np.zeros((2, 28, 28, 3))
    labels = np.zeros(2)

    results = engine.run(images, ground_truth=labels, verbose=False)

    assert len(results) == 2
    assert isinstance(results[0], ExperimentResult)
    assert len(results[0].attacks) == 1
    # Successful? Pred=0. GT=0. OriginalPred=0.
    # AdvPred=0. Success = (0!=0) = False (Untargeted)
    assert results[0].attacks[0].success == False

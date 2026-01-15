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
    assert not results[0].attacks[0].success


# Tests trying the callback autowrapper metric (input as a function)
def test_engine_with_callback_metric():
    clf = MockClassifier()
    art_atk = MockArtAttack(clf)
    adapter = ArtAttackAdapter(art_atk, name="MockAttack")

    def custom_metric(original, adversarial):
        return np.abs(original - adversarial).astype(np.uint8)

    engine_custom = AdversarialEngine(clf, [adapter], custom_metric)
    engine_mae = AdversarialEngine(clf, [adapter], MAEMetric())

    images = np.zeros((2, 28, 28, 3))
    labels = np.zeros(2)
    results_custom = engine_custom.run(images, ground_truth=labels, verbose=False)
    results_mae = engine_mae.run(images, ground_truth=labels, verbose=False)

    assert len(results_custom) == len(results_mae)
    for res_cust, res_mae in zip(results_custom, results_mae):
        diff_cust = res_cust.attacks[0].adversarial_image
        diff_mae = res_mae.attacks[0].adversarial_image
        assert np.array_equal(diff_cust, diff_mae)
        pred_custom = res_cust.attacks[0].prediction
        pred_mae = res_mae.attacks[0].prediction

        assert pred_custom == pred_mae
        
def test_engine_with_custom_metric_object():
    clf = MockClassifier()
    art_atk = MockArtAttack(clf)
    adapter = ArtAttackAdapter(art_atk, name="MockAttack")

    class CustomMetric:
        def calculate(self, original, adversarial):
            return np.abs(original - adversarial).astype(np.uint8)

    custom_metric = CustomMetric()
    engine_custom = AdversarialEngine(clf, [adapter], custom_metric)

    images = np.zeros((2, 28, 28, 3))
    labels = np.zeros(2)
    results_custom = engine_custom.run(images, ground_truth=labels, verbose=False)

    assert len(results_custom) == 2
    for res in results_custom:
        diff = res.attacks[0].adversarial_image
        assert np.array_equal(diff, np.zeros((28, 28, 3), dtype=np.uint8))
        pred = res.attacks[0].prediction
        assert pred == 0

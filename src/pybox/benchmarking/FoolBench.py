# Imports
from typing import List, Optional

from pybox.core.models import BenchmarkResult, PointsResult
from ..core.interfaces import BenchmarkingStrategy, MetricStrategy
from ..attacks.art_adapter import ArtAttackAdapter
import numpy as np

from tqdm import tqdm


# Classes
class FoolBench:
    def __init__(self, attacks: List[ArtAttackAdapter]) -> None:
        self._attacks = attacks

    def __call__(
        self,
        image_np: np.ndarray,
        ground_truth: int,
        max_iters_range: List[int],
        target: Optional[int] = None,
    ) -> BenchmarkResult:
        # Optimize the order for warm starts
        max_iters_range = sorted(max_iters_range)
        results = BenchmarkResult(points_results=[])
        # For each attack
        for attack in (pb := tqdm(self._attacks)):
            # Try each max iteration setting
            attack_points = PointsResult(
                method_name=attack.name,
                points=[],
            )
            for iterations in max_iters_range:
                attack._attack.__setattr__("max_iter", iterations)
                pb.set_description(f"Iteration for {attack.name} now {iterations}")
                # Generate adversarial example
                adversarial_example = attack.generate(image_np, target=target)
                predictions = attack._attack.estimator.predict(
                    adversarial_example[np.newaxis, ...].astype(np.float32)
                )

                attack_points.points.append((iterations, predictions[0][ground_truth]))

            results.points_results.append(attack_points)

        return results

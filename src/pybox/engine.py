"""
Adversarial Engine.
Orchestrates the generation of adversarial examples and checking of metrics.
"""

from typing import List, Optional, Any
import numpy as np
from tqdm import tqdm

from .core.interfaces import AttackStrategy, MetricStrategy
from .core.models import ExperimentResult, AttackResult


class AdversarialEngine:
    """
    Engine to run adversarial attacks and compute metrics.
    """

    def __init__(
        self,
        classifier: Any,
        attacks: List[AttackStrategy],
        metric: MetricStrategy,
        clip_values: tuple[float, float] | None = (0.0, 1.0),
    ):
        """
        Initialize the engine.

        Args:
            classifier: The classifier to attack (must have predict method).
            attacks: List of attack strategies.
            metric: Metric strategy to evaluate differences.
            clip_values: Range to clip images (min, max).
        """
        self.classifier = classifier
        self.attacks = attacks
        self.metric = metric
        self.clip_values = clip_values

    def run(
        self,
        images: np.ndarray,
        ground_truth: np.ndarray | None = None,
        target_labels: np.ndarray | None = None,
        verbose: bool = True,
    ) -> List[ExperimentResult]:
        """
        Run the experiment on a batch of images.

        Args:
            images: Input images (N, H, W, C).
            ground_truth: Ground truth labels (N,).
            target_labels: Target labels for attacks (N,).
            verbose: Whether to show progress bar.

        Returns:
            List of ExperimentResult objects.
        """
        results = []
        iterator = range(len(images))
        if verbose:
            iterator = tqdm(iterator, desc="Processing images")

        for i in iterator:
            image = images[i]
            gt = ground_truth[i] if ground_truth is not None else None
            target = target_labels[i] if target_labels is not None else None

            result = self._process_single_image(image, gt, target)
            results.append(result)

        return results

    def _process_single_image(
        self, image: np.ndarray, ground_truth: Optional[int], target: Optional[int]
    ) -> ExperimentResult:
        # Clip original if needed
        if self.clip_values:
            image = np.clip(image, self.clip_values[0], self.clip_values[1])

        # Get original prediction
        # Expecting classifier to take batch (1, H, W, C)
        probs = self.classifier.predict(image[np.newaxis, ...])
        orig_pred = int(np.argmax(probs, axis=1)[0])

        attack_results = []

        for attack in self.attacks:
            # Generate adversarial
            adv_img = attack.generate(image, target)

            # Clip adversarial
            if self.clip_values:
                adv_img = np.clip(adv_img, self.clip_values[0], self.clip_values[1])

            # Ensure dtype is float32 for PyTorch compatibility
            adv_img = adv_img.astype(np.float32)

            # Predict on adversarial
            adv_probs = self.classifier.predict(adv_img[np.newaxis, ...])
            adv_pred = int(np.argmax(adv_probs, axis=1)[0])

            # Check success
            if target is not None:
                success = adv_pred == target
            else:
                # Untargeted: success if prediction changed
                # If ground truth is known, success implies misclassification relative to GT?
                # Or relative to original prediction?
                # Standard definition: misclassified.
                if ground_truth is not None:
                    success = adv_pred != ground_truth
                else:
                    success = adv_pred != orig_pred

            # Compute metric
            # Only compute if we want to? Original code computed if success or always?
            # Original code: if success: compute difference. else: zeros.
            # We will compute it always or handle it.
            # Ideally we want to know the difference regardless of success?
            # But let's follow the "success" logic if it saves time, or just compute always.
            # Original code said: "if success: difference... else pass"

            diff_map = None
            if success:
                diff_map = self.metric.calculate(image, adv_img)

            attack_res = AttackResult(
                method_name=attack.name,
                adversarial_image=adv_img,
                prediction=adv_pred,
                success=success,
                difference_map=diff_map,
            )
            attack_results.append(attack_res)

        return ExperimentResult(
            original_image=image,
            ground_truth=ground_truth,
            original_prediction=orig_pred,
            attacks=attack_results,
        )

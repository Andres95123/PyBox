"""
Adversarial Engine.
Orchestrates the generation of adversarial examples and checking of metrics.
"""

from typing import List, Optional, Any, Callable, Literal
import numpy as np
from tqdm import tqdm

from .core.interfaces import AttackStrategy, MetricStrategy
from .core.models import ExperimentResult, AttackResult
from .metrics import METRIC_MAP, METRIC_LITERALS


class AdversarialEngine:
    """
    Engine to run adversarial attacks and compute metrics.
    """

    def __init__(
        self,
        classifier: Any,
        attacks: List[AttackStrategy],
        metric: MetricStrategy
        | Callable[[np.ndarray, np.ndarray], np.ndarray]
        | METRIC_LITERALS,
        clip_values: tuple[float, float] | None = (0.0, 1.0),
    ):
        """
        Initialize the engine.

        Args:
            classifier: The classifier to attack (must have predict method).
            attacks: List of attack strategies.
            metric: Metric strategy to evaluate differences. If a callable is provided, it will be wrapped automatically.
            clip_values: Range to clip images (min, max).
        """
        self.classifier = classifier
        self.attacks = attacks

        # Support callable metrics by wrapping and also string literals for basic metrics
        if isinstance(metric, str):
            if metric in METRIC_MAP:
                self.metric = METRIC_MAP[metric]()
            else:
                raise ValueError(f"Unknown metric: {metric}")
        elif callable(metric) and not isinstance(metric, MetricStrategy):
            # Wrap callable into MetricStrategy
            class CallableMetricAdapter:
                def __init__(self, func):
                    self.func = func

                def __call__(self, original, adversarial):
                    return self.func(original, adversarial)

            self.metric = CallableMetricAdapter(metric)
        else:
            self.metric = metric

        self.clip_values = clip_values

    def run(
        self,
        images: np.ndarray,
        ground_truth: np.ndarray | None = None,
        target_labels: np.ndarray | None = None,
        verbose: bool = True,
        batch_size: int = 32,
    ) -> List[ExperimentResult]:
        """
        Run the experiment on a batch of images.

        Args:
            images: Input images (N, H, W, C).
            ground_truth: Ground truth labels (N,).
            target_labels: Target labels for attacks (N,).
            verbose: Whether to show progress bar.
            batch_size: Number of images to process at once. Default 32 for performance.

        Returns:
            List of ExperimentResult objects.
        """
        results = []
        total_images = len(images)
        pbar = None

        if verbose:
            pbar = tqdm(total=total_images, desc="Processing images")

        try:
            for i in range(0, total_images, batch_size):
                batch_end = min(i + batch_size, total_images)

                batch_images = images[i:batch_end]
                batch_gt = (
                    ground_truth[i:batch_end] if ground_truth is not None else None
                )
                batch_targets = (
                    target_labels[i:batch_end] if target_labels is not None else None
                )

                batch_results = self._process_batch(
                    batch_images, batch_gt, batch_targets
                )
                results.extend(batch_results)

                if pbar:
                    pbar.update(batch_end - i)
        finally:
            if pbar:
                pbar.close()

        return results

    def _process_batch(
        self,
        images: np.ndarray,
        ground_truth: np.ndarray | None,
        targets: np.ndarray | None,
    ) -> List[ExperimentResult]:
        """
        Process a batch of images efficiently.
        """
        # Batch size for this chunk
        N = len(images)

        # Clip original if needed (Batch op)
        if self.clip_values:
            images = np.clip(images, self.clip_values[0], self.clip_values[1])

        # Get original predictions (Batch op)
        # Assuming classifier has predict method accepting (N, H, W, C)
        probs = self.classifier.predict(images)
        orig_preds = np.argmax(probs, axis=1)

        # Helper to hold attack data
        processed_attacks = []

        for attack in self.attacks:
            # Generate adversarial (Batch op)
            adv_imgs = attack.generate(images, targets)

            # Clip (Batch op)
            if self.clip_values:
                adv_imgs = np.clip(adv_imgs, self.clip_values[0], self.clip_values[1])

            # Ensure dtype is float32 for compatibility
            adv_imgs = adv_imgs.astype(np.float32)

            # Predict (Batch op)
            adv_probs = self.classifier.predict(adv_imgs)
            adv_preds = np.argmax(adv_probs, axis=1)

            # Check success (Vectorized)
            if targets is not None:
                successes = adv_preds == targets
            else:
                if ground_truth is not None:
                    successes = adv_preds != ground_truth
                else:
                    successes = adv_preds != orig_preds

            processed_attacks.append(
                {
                    "attack": attack,
                    "adv_imgs": adv_imgs,
                    "preds": adv_preds,
                    "successes": successes,
                }
            )

        # Assemble results per image to compute metrics (Loop)
        # We loop here to use per-image metric normalization as per legacy behavior
        batch_experiment_results = []

        for i in range(N):
            image = images[i]
            gt = ground_truth[i] if ground_truth is not None else None
            orig_pred = int(orig_preds[i])

            attack_results = []

            for p_attack in processed_attacks:
                attack = p_attack["attack"]
                adv_img = p_attack["adv_imgs"][i]
                adv_pred = int(p_attack["preds"][i])
                success = bool(p_attack["successes"][i])

                diff_map = None
                if success:
                    # Metric computation is per-image to preserve normalization behavior
                    diff_map = self.metric(image, adv_img)

                attack_results.append(
                    AttackResult(
                        method_name=attack.name,
                        adversarial_image=adv_img,
                        prediction=adv_pred,
                        success=success,
                        difference_map=diff_map,
                    )
                )

            batch_experiment_results.append(
                ExperimentResult(
                    original_image=image,
                    ground_truth=gt,
                    original_prediction=orig_pred,
                    attacks=attack_results,
                )
            )

        return batch_experiment_results

    def _process_single_image(
        self, image: np.ndarray, ground_truth: Optional[int], target: Optional[int]
    ) -> ExperimentResult:
        # Compatibility wrapper for single images
        img_batch = image[np.newaxis, ...]
        gt_batch = np.array([ground_truth]) if ground_truth is not None else None
        target_batch = np.array([target]) if target is not None else None

        return self._process_batch(img_batch, gt_batch, target_batch)[0]

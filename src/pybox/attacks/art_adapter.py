"""
Adapter for ART (Adversarial Robustness Toolbox) attacks.
"""

from typing import Optional, Any
import numpy as np
from ..core.interfaces import AttackStrategy


class ArtAttackAdapter:
    """
    Adapter to make ART attacks compatible with AttackStrategy protocol.
    """

    def __init__(self, art_attack: Any, name: Optional[str] = None):
        """
        Initialize the adapter.

        Args:
            art_attack: An instance of an ART attack (or compatible object).
            name: Optional name for the attack. Defaults to class name.
        """
        self._attack = art_attack
        self.name = name or art_attack.__class__.__name__

    def generate(
        self, image: np.ndarray, target: int | np.ndarray | None = None
    ) -> np.ndarray:
        """
        Generate an adversarial example using the wrapped ART attack.

        Args:
            image: Original image (H, W, C) or Batch (N, H, W, C).
            target: Target label index or array of indices (optional).

        Returns:
            Adversarial image (H, W, C) or Batch (N, H, W, C).
        """
        # Determine if input is a batch
        is_batch = image.ndim == 4

        # Expand dims for batch processing if single image (ART expects batches)
        x = image if is_batch else image[np.newaxis, ...]

        y = None
        if target is not None:
            # Ensure target is an array matching the batch size
            y = np.atleast_1d(target)

        adv_x = self._attack.generate(x=x, y=y)

        # Return in the same dimensionality as input
        return adv_x if is_batch else adv_x[0]

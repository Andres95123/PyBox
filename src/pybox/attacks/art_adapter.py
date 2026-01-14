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

    def generate(self, image: np.ndarray, target: int | None = None) -> np.ndarray:
        """
        Generate an adversarial example using the wrapped ART attack.

        Args:
            image: Original image (H, W, C).
            target: Target label index (optional).

        Returns:
            Adversarial image (H, W, C).
        """
        # Expand dims for batch processing (ART expects batches)
        x = image[np.newaxis, ...]

        if target is not None:
            y = np.array([target])
            adv_x = self._attack.generate(x=x, y=y)
        else:
            adv_x = self._attack.generate(x=x)

        return adv_x[0]

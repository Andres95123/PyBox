"""
Core interfaces for PyBox using Protocols.
This file defines the contracts that components must satisfy.
"""

from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class AttackStrategy(Protocol):
    """
    Interface for any adversarial attack.
    """

    def generate(self, image: np.ndarray, target: int | None = None) -> np.ndarray:
        """
        Generates an adversarial example.

        Args:
            image: The original image.
            target: The target label (optional).

        Returns:
            The adversarial image.
        """
        ...


@runtime_checkable
class MetricStrategy(Protocol):
    """
    Interface for comparing two images.
    """

    def __call__(self, original: np.ndarray, adversarial: np.ndarray) -> np.ndarray:
        """
        Calculates the difference between two images.

        Args:
            original: The original image.
            adversarial: The adversarial image.

        Returns:
            A difference map (image).
        """
        ...


@runtime_checkable
class BenchmarkingStrategy(Protocol):
    """
    Interface for benchmarking testing.
    """
    ...
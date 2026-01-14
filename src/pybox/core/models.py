"""
Core data models for PyBox.
This file defines the data structures used to pass information between components.
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np


@dataclass
class AttackResult:
    """
    Result of a single adversarial attack.
    """

    method_name: str
    adversarial_image: np.ndarray
    prediction: int
    success: bool
    difference_map: Optional[np.ndarray] = None


@dataclass
class ExperimentResult:
    """
    Result of the entire experiment on a single image.
    """

    original_image: np.ndarray
    ground_truth: Optional[int]
    original_prediction: int
    attacks: List[AttackResult]

"""
Data models for the GWO Clothing Production Optimizer.
"""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass
class ProductionProblem:
    """
    Stores all input data for a production planning optimization problem.
    """

    products: List[str]
    profits: np.ndarray
    pollution: np.ndarray
    time: np.ndarray
    resources: np.ndarray
    demand: np.ndarray
    max_time: float
    max_resources: float
    pollution_weight: float = 50.0
    constraint_penalty: float = 1000.0

    @property
    def dimensions(self) -> int:
        return len(self.products)


@dataclass
class OptimizationResult:
    """
    Stores the optimizer output.
    """

    solution: np.ndarray
    profit: float
    pollution: float
    time: float
    resources: float
    fitness: float
    constraint_status: Dict[str, bool]
    histories: Dict[str, Dict[str, list]]

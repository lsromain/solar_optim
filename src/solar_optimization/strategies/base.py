from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Tuple
import numpy as np

from ..devices.cet import CETProperties
from ..metrics.models import OptimizationMetrics
from ..metrics.calculator import MetricsCalculator

class OptimizationStrategy(ABC):
    def __init__(self, name: str):
        self.name = name
        self.metrics_calculator = MetricsCalculator()

    @abstractmethod
    def optimize(self, timestamps: List[datetime], 
                solar_production: np.ndarray,
                base_consumption: np.ndarray, 
                cet_properties: CETProperties) -> np.ndarray:
        pass

    def run_optimization(self, timestamps: List[datetime],
                        solar_production: np.ndarray,
                        base_consumption: np.ndarray,
                        cet_properties: CETProperties) -> Tuple[np.ndarray, OptimizationMetrics]:
        cet_consumption = self.optimize(timestamps, solar_production,
                                      base_consumption, cet_properties)
        metrics = self.metrics_calculator.run(timestamps, solar_production,
                                                  base_consumption, cet_consumption)
        return cet_consumption, metrics
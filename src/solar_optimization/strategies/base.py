from abc import ABC, abstractmethod
from typing import Tuple
from dataclasses import dataclass
import numpy as np

from ..devices.cet import CETProperties
from ..metrics.models import OptimizationMetrics
from ..metrics.calculator import MetricsCalculator
from ..core.scenarios import Scenario

@dataclass
class OptimizationResult():
    cet_consumption: np.ndarray
    metrics: OptimizationMetrics

class OptimizationStrategy(ABC):
    def __init__(self, name: str):
        self.name = name
        self.metrics_calculator = MetricsCalculator()

    @abstractmethod
    def optimize(self, scenario:Scenario, 
                cet_properties: CETProperties) -> np.ndarray:
        pass

    def run_optimization(self, scenario:Scenario,
                        cet_properties: CETProperties) -> OptimizationResult:
        cet_consumption = self.optimize(scenario, cet_properties)
        metrics = self.metrics_calculator.run(scenario, cet_consumption)
        return OptimizationResult(cet_consumption, metrics)


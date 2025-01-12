from datetime import datetime
from typing import List, Dict
import numpy as np

from solar_optimization.strategies.base import OptimizationStrategy
from solar_optimization.devices.cet import CETProperties
from solar_optimization.core.scenarios import Scenario

class ScheduledStrategy(OptimizationStrategy):
    """
    This strategy aims to consume based on a static schedule:
        * Stop condition trigger: Schedule is off at this timestamp
        * Start condition trigger: Schedule is on at this timestamp
    """
    def __init__(self, name: str, schedules: List[Dict[str, datetime]]):
        super().__init__(name)
        self.schedules = schedules

    def optimize(self, scenario:Scenario, cet_properties: CETProperties) -> np.ndarray:
        cet_consumption = np.zeros_like(scenario.timestamps)
        for schedule in self.schedules:
            cet_mask = [(t >= schedule["start"]) & (t < schedule["end"]) for t in scenario.timestamps]
            cet_consumption[cet_mask] = cet_properties.power
        return cet_consumption

from datetime import datetime
from typing import List, Dict
import numpy as np

from .base import OptimizationStrategy
from ..devices.cet import CETProperties

class ScheduledStrategy(OptimizationStrategy):
    def __init__(self, name: str, schedules: List[Dict[str, datetime]]):
        super().__init__(name)
        self.schedules = schedules

    def optimize(self, timestamps: List[datetime], solar_production: np.ndarray,
                base_consumption: np.ndarray, cet_properties: CETProperties) -> np.ndarray:
        cet_consumption = np.zeros_like(timestamps)
        for schedule in self.schedules:
            cet_mask = [(t >= schedule["start"]) & (t < schedule["end"]) for t in timestamps]
            cet_consumption[cet_mask] = cet_properties.power
        return cet_consumption
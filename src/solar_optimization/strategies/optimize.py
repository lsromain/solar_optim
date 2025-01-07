from datetime import datetime, timedelta
from typing import List
import numpy as np

from .base import OptimizationStrategy
from ..devices.cet import CETProperties
from ..core.scenarios import Scenario

class OptimizationStrategy(OptimizationStrategy):
    def __init__(self, name: str, threshold: float):
        super().__init__(name)
        self.threshold = threshold

    def optimize(self, scenario:Scenario, cet_properties: CETProperties) -> np.ndarray:
        timestamps = scenario.timestamps
        base_consumption = scenario.consumption_data
        solar_production = scenario.production_data
        cet_consumption = np.zeros_like(timestamps)
        
        state_duration = timedelta(minutes=0)
        total_running_duration = timedelta(minutes=0)
        state_init_timestamp = timestamps[0]
        is_running = False

        for i in range(len(timestamps)):
            state_duration = timestamps[i] - state_init_timestamp
            self_consumption_ratio_with_cet = abs(solar_production[i])/(base_consumption[i] + cet_properties.power)

            if is_running:
                if (total_running_duration + state_duration) >= cet_properties.max_duration:
                    total_running_duration = cet_properties.max_duration
                    break

                if (self_consumption_ratio_with_cet <= self.threshold and 
                    state_duration >= cet_properties.min_duration):
                    total_running_duration += state_duration
                    is_running = False
                    state_init_timestamp = timestamps[i]
                else:
                    cet_consumption[i] = cet_properties.power
            else:
                if (self_consumption_ratio_with_cet >= self.threshold and 
                    state_duration >= cet_properties.min_duration):
                    is_running = True
                    state_init_timestamp = timestamps[i]
                    cet_consumption[i] = cet_properties.power

        state_duration = timedelta(minutes=0)
        state_end_timestamp = timestamps[-1]
        i = 1
        while (total_running_duration + state_duration) < cet_properties.max_duration:
            cet_consumption[-i] = cet_properties.power
            i += 1
            state_duration = state_end_timestamp - timestamps[-(i)]

        return cet_consumption
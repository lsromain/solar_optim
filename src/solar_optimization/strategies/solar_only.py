from datetime import datetime, timedelta
from typing import List
import numpy as np

from .base import OptimizationStrategy
from ..devices.cet import CETProperties

class SolarOnlyStrategy(OptimizationStrategy):
    def __init__(self, name: str, threshold_start: float):
        super().__init__(name)
        self.threshold_start = threshold_start

    def optimize(self, timestamps: List[datetime], solar_production: np.ndarray,
                base_consumption: np.ndarray, cet_properties: CETProperties) -> np.ndarray:
        cet_consumption = np.zeros_like(timestamps)
        threshold_start = cet_properties.power * self.threshold_start
        grid_exchange = base_consumption - solar_production
        
        state_duration = timedelta(minutes=0)
        total_running_duration = timedelta(minutes=0)
        state_init_timestamp = timestamps[0]
        is_running = False

        for i in range(len(timestamps)):
            state_duration = timestamps[i] - state_init_timestamp
            
            if is_running:
                power_from_grid = grid_exchange[i] + cet_properties.power
                
                if (total_running_duration + state_duration) >= cet_properties.max_duration:
                    break

                if power_from_grid > 0 and state_duration >= cet_properties.min_duration:
                    total_running_duration += state_duration
                    is_running = False
                    state_init_timestamp = timestamps[i]
                else:
                    cet_consumption[i] = cet_properties.power
            else:
                available_solar_power = -grid_exchange[i]
                if available_solar_power >= threshold_start and state_duration >= cet_properties.min_duration:
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
from datetime import datetime, timedelta
from typing import List
import numpy as np

from .base import OptimizationStrategy
from ..devices.cet import CETProperties

class MaximizeSolarStrategy(OptimizationStrategy):
    def __init__(self, name: str):
        super().__init__(name)

    def optimize(self, timestamps: List[datetime], solar_production: np.ndarray,
                base_consumption: np.ndarray, cet_properties: CETProperties) -> np.ndarray:
        cet_consumption = np.zeros_like(timestamps)
        grid_exchange = base_consumption - solar_production
        
        state_duration = timedelta(minutes=0)
        total_running_duration = timedelta(minutes=0)
        state_init_timestamp = timestamps[0]
        is_running = False

        for i in range(len(timestamps)):
            state_duration = timestamps[i] - state_init_timestamp
            power_from_grid_without_cet = grid_exchange[i]

            if is_running:
                if (total_running_duration + state_duration) >= cet_properties.max_duration:
                    total_running_duration = cet_properties.max_duration
                    break

                if power_from_grid_without_cet > 0 and state_duration >= cet_properties.min_duration:
                    total_running_duration += state_duration
                    is_running = False
                    state_init_timestamp = timestamps[i]
                else:
                    cet_consumption[i] = cet_properties.power
            else:
                available_solar_power = -power_from_grid_without_cet
                if available_solar_power >= 0 and state_duration >= cet_properties.min_duration:
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
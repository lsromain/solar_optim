from datetime import timedelta
import numpy as np

from solar_optimization.strategies.base import OptimizationStrategy
from solar_optimization.devices.cet import CETProperties
from solar_optimization.core.scenarios import Scenario

class SolarOnlyStrategy(OptimizationStrategy):
    """
    This strategy aims to consume only if the CET consumption can be 100% covered by the solar production:
        * Stop condition trigger: When starting to draw energy from the grid
        * Start condition trigger: available solar production covers (100+xx)% of CET consumption
    """
    def __init__(self, name: str, threshold_start: float):
        super().__init__(name)
        self.threshold_start = threshold_start

    def optimize(self, scenario:Scenario, cet_properties: CETProperties) -> np.ndarray:
        
        cet_consumption = np.zeros_like(scenario.timestamps)
        threshold_start = cet_properties.power * self.threshold_start
        grid_exchange = scenario.consumption_data - scenario.production_data
        
        state_duration = timedelta(minutes=0)
        total_running_duration = timedelta(minutes=0)
        state_init_timestamp = scenario.timestamps[0]
        is_running = False

        for i in range(len(scenario.timestamps)):
            state_duration = scenario.timestamps[i] - state_init_timestamp
            
            if is_running:
                power_from_grid = grid_exchange[i] + cet_properties.power
                
                if (total_running_duration + state_duration) >= cet_properties.max_duration:
                    total_running_duration = cet_properties.max_duration
                    break

                # Stop condition trigger: When starting to draw energy from the grid
                if power_from_grid > 0 and state_duration >= cet_properties.min_duration:
                    total_running_duration += state_duration
                    is_running = False
                    state_init_timestamp = scenario.timestamps[i]
                else:
                    cet_consumption[i] = cet_properties.power
            else:
                available_solar_power = -grid_exchange[i]
                # Start condition trigger: available solar production covers (100+xx)% of CET consumption
                if available_solar_power >= threshold_start and state_duration >= cet_properties.min_duration:
                    is_running = True
                    state_init_timestamp = scenario.timestamps[i]
                    cet_consumption[i] = cet_properties.power
        
        state_duration = timedelta(minutes=0)
        state_end_timestamp = scenario.timestamps[-1]
        i = 1
        while (total_running_duration + state_duration) < cet_properties.max_duration:
            cet_consumption[-i] = cet_properties.power
            i += 1
            state_duration = state_end_timestamp - scenario.timestamps[-(i)]

        return cet_consumption
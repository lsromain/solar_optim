from datetime import timedelta
import numpy as np

from solar_optimization.strategies.base import OptimizationStrategy
from solar_optimization.devices.cet import CETProperties
from solar_optimization.core.scenarios import Scenario

class MaximizeSolarStrategy(OptimizationStrategy):
    """
    This strategy aims at consuming as much available solar energy as possible:
        * Stop condition trigger: Without CET, the home would consume 100% of the solar production
        * Start condition trigger: When injecting power to the grid
    """
    def __init__(self, name: str):
        super().__init__(name)

    def optimize(self, scenario:Scenario, cet_properties: CETProperties) -> np.ndarray:
        
        cet_consumption = np.zeros_like(scenario.timestamps)
        grid_exchange = scenario.consumption_data - scenario.production_data
        
        state_duration = timedelta(minutes=0)
        total_running_duration = timedelta(minutes=0)
        state_init_timestamp = scenario.timestamps[0]
        is_running = False

        for i in range(len(scenario.timestamps)):
            state_duration = scenario.timestamps[i] - state_init_timestamp
            power_from_grid_without_cet = grid_exchange[i]

            if is_running:
                if (total_running_duration + state_duration) >= cet_properties.max_duration:
                    total_running_duration = cet_properties.max_duration
                    break
                
                # Stop condition trigger: Without CET, the home would consume 100% of the solar production
                if power_from_grid_without_cet > 0 and state_duration >= cet_properties.min_duration:
                    total_running_duration += state_duration
                    is_running = False
                    state_init_timestamp = scenario.timestamps[i]
                else:
                    cet_consumption[i] = cet_properties.power
            else:
                available_solar_power = -power_from_grid_without_cet
                # Start condition trigger: When starting injecting in the grid
                if available_solar_power >= 0 and state_duration >= cet_properties.min_duration:
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
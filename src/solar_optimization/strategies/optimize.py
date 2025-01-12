from datetime import timedelta
import numpy as np

from solar_optimization.strategies.base import OptimizationStrategy
from solar_optimization.devices.cet import CETProperties
from solar_optimization.core.scenarios import Scenario

class OptimizationStrategy(OptimizationStrategy):
    """
    This strategy aims to consumer when the home consumption ratio is high enough:
        * Stop condition trigger: With CET, the home consumption ratio would be lower than the threshold
        * Start condition trigger: With CET, the home consumption ratio would be higher than the threshold
    """
    def __init__(self, name: str, threshold: float):
        super().__init__(name)
        self.threshold = threshold

    def optimize(self, scenario:Scenario, cet_properties: CETProperties) -> np.ndarray:
        cet_consumption = np.zeros_like(scenario.timestamps)
        
        state_duration = timedelta(minutes=0)
        total_running_duration = timedelta(minutes=0)
        state_init_timestamp = scenario.timestamps[0]
        is_running = False

        for i in range(len(scenario.timestamps)):
            state_duration = scenario.timestamps[i] - state_init_timestamp
            self_consumption_ratio_with_cet = abs(scenario.production_data[i])/(scenario.consumption_data[i] + cet_properties.power)

            if is_running:
                if (total_running_duration + state_duration) >= cet_properties.max_duration:
                    total_running_duration = cet_properties.max_duration
                    break

                # Stop condition trigger: With CET, the home consumption ratio would be lower than the threshold
                if (self_consumption_ratio_with_cet <= self.threshold and 
                    state_duration >= cet_properties.min_duration):
                    total_running_duration += state_duration
                    is_running = False
                    state_init_timestamp = scenario.timestamps[i]
                else:
                    cet_consumption[i] = cet_properties.power
            else:
                # Start condition trigger: With CET, the home consumption ratio would be higher than the threshold
                if (self_consumption_ratio_with_cet >= self.threshold and 
                    state_duration >= cet_properties.min_duration):
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
from dataclasses import dataclass
import numpy as np

from .timeseries import TimeSeriesConfig
from .consumption import BaseConsumption
from .production import SolarProduction

@dataclass
class ScenarioInputs:
    timestamps: np.ndarray       #List[dateTime]
    base_consumption: BaseConsumption #kW
    solar_production: SolarProduction #kW
    
class Scenario:
    def __init__(self, scenario_data:ScenarioInputs):
        self.consumption_data = scenario_data.base_consumption.generate(scenario_data.timestamps)
        self.production_data = scenario_data.solar_production.generate(scenario_data.timestamps)
        self.timestamps = scenario_data.timestamps
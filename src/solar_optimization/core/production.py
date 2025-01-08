from datetime import datetime, timedelta
from typing import List
from enum import Enum
import numpy as np

class SolarProductionPeak:
    def __init__(self, peak: float, start: datetime, end: datetime):
        self.peak = peak
        self.start = start
        self.end = end

class SolarProduction:
    def __init__(self, peaks: List[SolarProductionPeak]):
        self.peaks = peaks

    def generate(self, timestamps: List[datetime]) -> np.ndarray:
        t = np.linspace(0, (timestamps[-1]-timestamps[0])/timedelta(hours=1), len(timestamps))
        solar_production = np.zeros_like(t)
        
        for peak in self.peaks:
            mask = [(timestamp >= peak.start) & (timestamp <= peak.end) for timestamp in timestamps]
            t_normalized = (t[mask] - np.min(t[mask])) / (np.max(t[mask]) - np.min(t[mask]))
            t_peak = np.mean(t[mask])
            solar_production[mask] = solar_production[mask] + (
                peak.peak * np.sin(np.pi * t_normalized) * 
                np.exp(-((t[mask] - t_peak) ** 2) / 8)
            )
        
        return np.abs(solar_production)
    
class DefaultProductionScenario(Enum):
    SUMMER_SUNNY_ALL_DAY = 1
    SUMMER_CLOUDY =2
    WINTER_SUNNY_ALL_DAY = 3
    WINTER_CLOUDY = 4

class DefaultSolarProduction:
    @classmethod
    def generate(self, scenario:DefaultProductionScenario)->SolarProduction:
        if scenario == DefaultProductionScenario.SUMMER_SUNNY_ALL_DAY:
            return SolarProduction(
                [SolarProductionPeak(2.7,   datetime(2024, 1, 1, 8, 0),  datetime(2024, 1, 1, 20, 0))])
        elif scenario == DefaultProductionScenario.SUMMER_CLOUDY:
            return SolarProduction(
                [SolarProductionPeak(0.9,   datetime(2024, 1, 1, 8, 30), datetime(2024, 1, 1, 11, 30)),
                 SolarProductionPeak(2.2,   datetime(2024, 1, 1, 11, 0), datetime(2024, 1, 1, 14, 0)),
                 SolarProductionPeak(1.2,   datetime(2024, 1, 1, 13, 30), datetime(2024, 1, 1, 17, 0))])
        elif scenario == DefaultProductionScenario.WINTER_SUNNY_ALL_DAY:
            return SolarProduction(
                [SolarProductionPeak(1.,    datetime(2024, 1, 1, 9, 0),  datetime(2024, 1, 1, 17, 0))])
        elif scenario == DefaultProductionScenario.WINTER_CLOUDY:
            return SolarProduction(
                [SolarProductionPeak(0.4,   datetime(2024, 1, 1, 9, 30), datetime(2024, 1, 1, 11, 30)),
                 SolarProductionPeak(0.9,   datetime(2024, 1, 1, 11, 0), datetime(2024, 1, 1, 17, 0))])
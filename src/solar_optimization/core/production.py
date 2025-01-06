from datetime import datetime, timedelta
from typing import List
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
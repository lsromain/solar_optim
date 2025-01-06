from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List
import numpy as np
from scipy.signal import savgol_filter

@dataclass
class ConsumptionPeak:
    name: str
    peak: float
    start: datetime
    end: datetime

class BaseConsumption:
    def __init__(self, mean_base: float, peaks: list[ConsumptionPeak]):
        self.mean_base = mean_base
        self.peaks = peaks

    def generate(self, timestamps: List[datetime]) -> np.ndarray:
        t = np.linspace(0, (timestamps[-1]-timestamps[0])/timedelta(hours=1), len(timestamps))
        n_points = len(t)
        base_consumption = np.ones(n_points) * self.mean_base

        for peak in self.peaks:
            peak_consumption = np.zeros_like(timestamps)
            mask = [(timestamp >= peak.start) & (timestamp <= peak.end) for timestamp in timestamps]
            t_normalized = (t[mask] - np.min(t[mask])) / (np.max(t[mask]) - np.min(t[mask]))
            peak_consumption[mask] = peak.peak * np.sin(np.pi * t_normalized)
            base_consumption = base_consumption + peak_consumption

        # Add noise and smooth
        base_consumption += np.random.normal(0, 0.05, n_points)
        base_consumption = savgol_filter(base_consumption, window_length=5, polyorder=3)
        
        return np.abs(base_consumption)
from datetime import datetime
from typing import List
import matplotlib.pyplot as plt
import numpy as np

from ..metrics.models import OptimizationMetrics

class SolarOptimizationVisualizer:
    @staticmethod
    def plot_strategy_results(timestamps: List[datetime], base_consumption: np.ndarray,
                            solar_production: np.ndarray, cet_consumption: np.ndarray,
                            metrics: OptimizationMetrics, strategy_name: str,
                            ax: plt.Axes) -> None:
        """Plot results for a single strategy"""
        home_consumption = base_consumption + cet_consumption
        grid_exchanges = home_consumption + solar_production
        
        time_delta = timestamps[1] - timestamps[0]
        width = time_delta.total_seconds() / (3600 * 24)
        
        ax.bar(timestamps, home_consumption, width, label='Consommation', color='#FF6B6B', alpha=0.7)
        ax.bar(timestamps, solar_production, width, label='Production solaire', color='#4ECB71', alpha=0.7)
        ax.plot(timestamps, grid_exchanges, 'b-', label='Échanges réseau', linewidth=2)
        ax.plot(timestamps, metrics.cet_active, 'r-', label='CET actif', linewidth=2)
        
        ax.set_xlabel('Heure')
        ax.set_ylabel('Puissance (kW)')
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_title(f'Stratégie: {strategy_name}')
        ax.set_ylim(-3, 3)
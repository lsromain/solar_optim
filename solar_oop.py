from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Any, NamedTuple
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

@dataclass
class TimeSeriesConfig:
    start_time: datetime
    end_time: datetime
    time_delta: timedelta

    def create_timestamps(self) -> List[datetime]:
        timestamps = []
        current_time = self.start_time
        while current_time <= self.end_time:
            timestamps.append(current_time)
            current_time += self.time_delta
        return timestamps

class ConsumptionPeak:
    def __init__(self, name: str, peak: float, start: datetime, end: datetime):
        self.name = name
        self.peak = peak
        self.start = start
        self.end = end

class BaseConsumption:
    def __init__(self, mean_base: float, peaks: List[ConsumptionPeak]):
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
                -peak.peak * np.sin(np.pi * t_normalized) * 
                np.exp(-((t[mask] - t_peak) ** 2) / 8)
            )
        
        return -np.abs(solar_production)

@dataclass
class CETProperties:
    power: float
    min_duration: timedelta
    max_duration: timedelta

@dataclass
class OptimizationMetrics:
    import_from_grid: float  # kWh
    export_to_grid: float    # kWh
    self_consumption_rate: float  # %
    total_cost: float  # €
    average_cost_per_kwh: float  # €/kWh
    cet_cost: float  # €
    cet_solar_ratio: float  # %
    cet_runtime: float  # hours

class OptimizationStrategy:
    def __init__(self, name: str):
        self.name = name
        self.import_tariff = 0.25  # €/kWh
        self.export_tariff = 0.1269   # €/kWh

    def optimize(self, timestamps: List[datetime], solar_production: np.ndarray,
                base_consumption: np.ndarray, cet_properties: CETProperties) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement optimize method")

    def calculate_metrics(self, timestamps: List[datetime], solar_production: np.ndarray,
                         base_consumption: np.ndarray, cet_consumption: np.ndarray) -> OptimizationMetrics:
        ## Delta between 2 timestamps in hours
        dt_hours = (timestamps[1]-timestamps[0]).total_seconds() / 3600
        
        # Calculate total consumption
        home_consumption = base_consumption + cet_consumption
        grid_exchanges = home_consumption + solar_production
        
        # Calculate grid exchanges
        grid_imports = np.maximum(grid_exchanges, 0)
        grid_exports = np.maximum(-grid_exchanges, 0)
        total_grid_imports = np.sum(grid_imports) * dt_hours
        total_grid_exports = np.sum(grid_exports) * dt_hours
        
        # Calculate total production and consumption
        total_solar_production = -np.sum(solar_production) * dt_hours
        total_home_consumption = np.sum(home_consumption) * dt_hours
        
        # Calculate self-consumption
        consumption_from_local_prod = total_solar_production - total_grid_exports
        self_consumption_rate = (consumption_from_local_prod / total_solar_production * 100) if total_solar_production > 0 else 0
        
        # Calculate costs
        import_cost = total_grid_imports * self.import_tariff
        export_revenue = total_grid_exports * self.export_tariff
        total_cost = import_cost - export_revenue
        mean_cost_per_kwh = total_cost / total_home_consumption if total_home_consumption > 0 else 0
        
        # Calculate CET specific metrics
        consumption_from_grid_ratio = grid_imports / home_consumption
        cet_consumption_cost = consumption_from_grid_ratio * self.import_tariff * cet_consumption * dt_hours
        total_cet_cost = np.sum(cet_consumption_cost)
        
        # Calculate CET solar ratio
        cet_mask = cet_consumption > 0
        if np.sum(cet_mask) > 0:
            cet_solar_ratio = np.sum((1 - consumption_from_grid_ratio[cet_mask]) * 
                                   cet_consumption[cet_mask]) / np.sum(cet_consumption[cet_mask]) * 100
        else:
            cet_solar_ratio = 0
            
        # Calculate CET runtime
        cet_runtime = np.sum(cet_mask) * dt_hours
        
        return OptimizationMetrics(
            import_from_grid=total_grid_imports,
            export_to_grid=total_grid_exports,
            self_consumption_rate=self_consumption_rate,
            total_cost=total_cost,
            average_cost_per_kwh=mean_cost_per_kwh,
            cet_cost=total_cet_cost,
            cet_solar_ratio=cet_solar_ratio,
            cet_runtime=cet_runtime
        )

    def run_optimization(self, timestamps: List[datetime], solar_production: np.ndarray,
                        base_consumption: np.ndarray, cet_properties: CETProperties) -> tuple[np.ndarray, OptimizationMetrics]:
        """Run the optimization strategy and calculate associated metrics"""
        cet_consumption = self.optimize(timestamps, solar_production, base_consumption, cet_properties)
        metrics = self.calculate_metrics(timestamps, solar_production, base_consumption, cet_consumption)
        return cet_consumption, metrics

class ScheduledStrategy(OptimizationStrategy):
    def __init__(self, name: str, schedules: List[Dict[str, datetime]]):
        super().__init__(name)
        self.schedules = schedules

    def optimize(self, timestamps: List[datetime], solar_production: np.ndarray,
                base_consumption: np.ndarray, cet_properties: CETProperties) -> np.ndarray:
        cet_consumption = np.zeros_like(timestamps)
        for schedule in self.schedules:
            cet_mask = [(t >= schedule["start"]) & (t < schedule["end"]) for t in timestamps]
            cet_consumption[cet_mask] = cet_properties.power
        return cet_consumption

class SolarOnlyStrategy(OptimizationStrategy):
    def __init__(self, name: str, threshold_start: float):
        super().__init__(name)
        self.threshold_start = threshold_start

    def optimize(self, timestamps: List[datetime], solar_production: np.ndarray,
                base_consumption: np.ndarray, cet_properties: CETProperties) -> np.ndarray:
        cet_consumption = np.zeros_like(timestamps)
        threshold_start = cet_properties.power * self.threshold_start
        grid_exchange = base_consumption + solar_production
        
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
                if (available_solar_power >= threshold_start and 
                    state_duration >= cet_properties.min_duration):
                    is_running = True
                    state_init_timestamp = timestamps[i]
                    cet_consumption[i] = cet_properties.power

        # Complete remaining duration if needed
        state_duration = timedelta(minutes=0)
        state_end_timestamp = timestamps[-1]
        i = 1
        while (total_running_duration + state_duration) < cet_properties.max_duration:
            cet_consumption[-i] = cet_properties.power
            i += 1
            state_duration = state_end_timestamp - timestamps[-(i)]

        return cet_consumption

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
        ax.plot(timestamps, (cet_consumption > 0).astype(float), 'r-', label='CET actif', linewidth=2)
        
        ax.set_xlabel('Heure')
        ax.set_ylabel('Puissance (kW)')
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_title(f'Stratégie: {strategy_name}')
        ax.set_ylim(-3, 3)

def main():
    # Configuration
    time_config = TimeSeriesConfig(
        start_time=datetime(2024, 1, 1, 0, 0),
        end_time=datetime(2024, 1, 2, 0, 0),
        time_delta=timedelta(minutes=5)
    )
    timestamps = time_config.create_timestamps()

    # Base consumption setup
    consumption_peaks = [
        ConsumptionPeak("morning_peak",
                        2, 
                       datetime(2024, 1, 1, 7, 0),
                       datetime(2024, 1, 1, 7, 30)),
        ConsumptionPeak("evening_peak",
                        2,
                       datetime(2024, 1, 1, 19, 0),
                       datetime(2024, 1, 1, 19, 30)),
        ConsumptionPeak("Lunch",
                        1.5,
                        datetime(2024, 1, 1, 13, 0),
                        datetime(2024, 1, 1, 13, 30)),
        ConsumptionPeak("Washing machine",
                        1.5,
                        datetime(2024, 1, 1, 10, 30),
                        datetime(2024, 1, 1, 11, 0))
    ]
    base_consumption = BaseConsumption(mean_base=0.5, peaks=consumption_peaks)

    # Solar production setup
    solar_peaks = [
        SolarProductionPeak(1.4,
                           datetime(2024, 1, 1, 9, 0),
                           datetime(2024, 1, 1, 11, 30)),
        SolarProductionPeak(2,
                           datetime(2024, 1, 1, 11, 0),
                           datetime(2024, 1, 1, 17, 0))
    ]
    solar_production = SolarProduction(peaks=solar_peaks)

    # Generate time series
    base_consumption_data = base_consumption.generate(timestamps)
    solar_production_data = solar_production.generate(timestamps)

    # CET properties
    cet_properties = CETProperties(
        power=1,
        min_duration=timedelta(minutes=15),
        max_duration=timedelta(hours=4)
    )

    # Create and run strategies
    strategies = [
        ScheduledStrategy("Night Scheduling", [
            {"start": datetime(2024, 1, 1, 0, 0),
             "end": datetime(2024, 1, 1, 2, 0)}
        ]),
        SolarOnlyStrategy("Solar Only", threshold_start=1.1),
    ]

    # Create figure for plotting
    fig = plt.figure(figsize=(15, len(strategies) * 4))
    gs = plt.GridSpec(len(strategies), 1, hspace=0.4)

    results = []
    for idx, strategy in enumerate(strategies):
        # Run optimization and get results
        cet_consumption, metrics = strategy.run_optimization(
            timestamps, solar_production_data,
            base_consumption_data, cet_properties
        )
        
        # Plot results for this strategy
        ax = fig.add_subplot(gs[idx])
        SolarOptimizationVisualizer.plot_strategy_results(
            timestamps, base_consumption_data,
            solar_production_data, cet_consumption,
            metrics, strategy.name, ax
        )
        
        # Store results for comparison
        results.append((strategy.name, metrics))

    # Print comparison of metrics
    print("\nComparison of Strategies:")
    print("-" * 80)
    for name, metrics in results:
        print(f"\nStrategy: {name}")
        print(f"Import from grid: {metrics.import_from_grid:.2f} kWh")
        print(f"Self-consumption rate: {metrics.self_consumption_rate:.1f}%")
        print(f"Total cost: {metrics.total_cost:.2f}€")
        print(f"CET solar ratio: {metrics.cet_solar_ratio:.1f}%")
        print(f"CET runtime: {metrics.cet_runtime:.1f} hours")

    plt.show()

if __name__ == "__main__":
    main()
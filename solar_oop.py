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
    production_totale: float  # kWh
    consommation_totale: float  # kWh
    import_reseau: float  # kWh
    export_reseau: float  # kWh
    taux_autoconsommation: float  # %
    cout_total: float  # €
    cout_moyen_kwh: float  # €/kWh
    cout_import: float  # €
    revenu_export: float  # €
    temps_fonctionnement_cet: float  # hours
    cout_fonctionnement_cet: float  # €
    cet_active: np.ndarray  # binary array
    cet_solar_share: float  # %

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
        dt_hours = (timestamps[1]-timestamps[0]).total_seconds() / 3600
        
        # Calculate total consumption and production
        home_consumption = base_consumption + cet_consumption
        grid_exchanges = home_consumption + solar_production
        total_solar_production = -np.sum(solar_production) * dt_hours
        total_home_consumption = np.sum(home_consumption) * dt_hours
        
        # Calculate grid exchanges
        mask_export = grid_exchanges < 0
        mask_import = grid_exchanges > 0
        grid_exports = np.zeros_like(grid_exchanges)
        grid_imports = np.zeros_like(grid_exchanges)
        grid_exports[mask_export] = -grid_exchanges[mask_export]
        grid_imports[mask_import] = grid_exchanges[mask_import]
        total_grid_imports = np.sum(grid_imports) * dt_hours
        total_grid_exports = np.sum(grid_exports) * dt_hours
        
        # Calculate consumption ratios
        consumption_from_grid_ratio = grid_imports/home_consumption
        
        # Calculate self-consumption
        consumption_from_local_prod = total_solar_production - total_grid_exports
        self_consumption_rate = (consumption_from_local_prod / total_solar_production * 100) if total_solar_production > 0 else 0
        
        # Calculate costs
        import_cost = total_grid_imports * self.import_tariff
        export_revenue = total_grid_exports * self.export_tariff
        total_cost = import_cost - export_revenue
        mean_cost_per_kwh = total_cost / total_home_consumption if total_home_consumption > 0 else 0
        
        # Calculate CET specific metrics
        grid_kwh_price = consumption_from_grid_ratio * self.import_tariff
        cet_consumption_cost = grid_kwh_price * cet_consumption * dt_hours
        cet_total_cost = np.sum(cet_consumption_cost)
        
        cet_active = np.zeros_like(timestamps)
        cet_active[cet_consumption > 0] = 1
        
        cet_solar_ratio = np.sum((1-consumption_from_grid_ratio) * cet_consumption) / np.sum(cet_consumption) * 100 if np.sum(cet_consumption) > 0 else 0
        
        cet_runtime_hours = np.sum(cet_active) * dt_hours
        
        return OptimizationMetrics(
            production_totale=total_solar_production,
            consommation_totale=total_home_consumption,
            import_reseau=total_grid_imports,
            export_reseau=total_grid_exports,
            taux_autoconsommation=self_consumption_rate,
            cout_total=total_cost,
            cout_moyen_kwh=mean_cost_per_kwh,
            cout_import=import_cost,
            revenu_export=export_revenue,
            temps_fonctionnement_cet=cet_runtime_hours,
            cout_fonctionnement_cet=cet_total_cost,
            cet_active=cet_active,
            cet_solar_share=cet_solar_ratio
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
                if available_solar_power >= threshold_start and state_duration >= cet_properties.min_duration:
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

class MaximizeSolarStrategy(OptimizationStrategy):
    def __init__(self, name: str):
        super().__init__(name)

    def optimize(self, timestamps: List[datetime], solar_production: np.ndarray,
                base_consumption: np.ndarray, cet_properties: CETProperties) -> np.ndarray:
        cet_consumption = np.zeros_like(timestamps)
        grid_exchange = base_consumption + solar_production
        
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
    
class OptimizationStrategy(OptimizationStrategy):
    def __init__(self, name: str, threshold: float):
        super().__init__(name)
        self.threshold = threshold

    def optimize(self, timestamps: List[datetime], solar_production: np.ndarray,
                base_consumption: np.ndarray, cet_properties: CETProperties) -> np.ndarray:
        cet_consumption = np.zeros_like(timestamps)
        
        state_duration = timedelta(minutes=0)
        total_running_duration = timedelta(minutes=0)
        state_init_timestamp = timestamps[0]
        is_running = False

        for i in range(len(timestamps)):
            state_duration = timestamps[i] - state_init_timestamp
            self_consumption_ratio_with_cet = abs(solar_production[i])/(base_consumption[i] + cet_properties.power)

            if is_running:
                if (total_running_duration + state_duration) >= cet_properties.max_duration:
                    total_running_duration = cet_properties.max_duration
                    break

                if (self_consumption_ratio_with_cet <= self.threshold and 
                    state_duration >= cet_properties.min_duration):
                    total_running_duration += state_duration
                    is_running = False
                    state_init_timestamp = timestamps[i]
                else:
                    cet_consumption[i] = cet_properties.power
            else:
                if (self_consumption_ratio_with_cet >= self.threshold and 
                    state_duration >= cet_properties.min_duration):
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
        ScheduledStrategy("Programmé la nuit", [
            {"start": datetime(2024, 1, 1, 0, 0),
             "end": datetime(2024, 1, 1, 2, 0)}
        ]),
        ScheduledStrategy("Programmé en journée", [
            {"start": datetime(2024, 1, 1, 11, 0),
             "end": datetime(2024, 1, 1, 15, 0)}
        ]),
        SolarOnlyStrategy("100% Solaire", threshold_start=1.1),
        MaximizeSolarStrategy("Max. Solar."),
        OptimizationStrategy("Optimiz", threshold=0.5)
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
    print("\nComparaison des Stratégies:")
    print("-" * 80)
    for name, metrics in results:
        print(f"\nStratégie: {name}")
        print(f"Import réseau: {metrics.import_reseau:.2f} kWh")
        print(f"Export réseau: {metrics.export_reseau:.1f} kWh")
        print(f"Taux autoconsommation: {metrics.taux_autoconsommation:.1f}%")
        print(f"Coût total: {metrics.cout_total:.2f}€")
        print(f"Coût moyen/kWh: {metrics.cout_moyen_kwh:.3f}€/kWh")
        print(f"Coût CET: {metrics.cout_fonctionnement_cet:.2f}€")
        print(f"CET solar ratio: {metrics.cet_solar_share:.1f}%")
        print(f"Temps fonctionnement CET: {metrics.temps_fonctionnement_cet:.1f} heures")

    plt.show()

if __name__ == "__main__":
    main()
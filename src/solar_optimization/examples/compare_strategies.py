from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from solar_optimization.core.timeseries import TimeSeriesConfig
from solar_optimization.core.consumption import BaseConsumption, ConsumptionPeak
from solar_optimization.core.production import SolarProduction, SolarProductionPeak
from solar_optimization.devices.cet import CETProperties
from solar_optimization.strategies.schedule import ScheduledStrategy
from solar_optimization.strategies.solar_only import SolarOnlyStrategy
from solar_optimization.strategies.max_solar import MaximizeSolarStrategy
from solar_optimization.strategies.optimize import OptimizationStrategy
from solar_optimization.visualization.plotter import SolarOptimizationVisualizer

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
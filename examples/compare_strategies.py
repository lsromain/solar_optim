from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from solar_optimization.core import TimeSeriesConfig, BaseConsumption, ConsumptionPeak, SolarProduction, SolarProductionPeak, Scenario, ScenarioInputs
from solar_optimization.devices.cet import CETProperties
from solar_optimization.strategies import ScheduledStrategy, SolarOnlyStrategy, MaximizeSolarStrategy, OptimizationStrategy
from solar_optimization.visualization.plotter import SolarOptimizationVisualizer


def main():
    ## Configurations
    # Timestamps
    timestamps = TimeSeriesConfig.create_timestamps(time_delta=timedelta(minutes=5))

    # Base consumption setup
    consumption_peaks = [
        ConsumptionPeak("morning_peak",     2,  datetime(2024, 1, 1, 7, 0),     datetime(2024, 1, 1, 7, 30)),
        ConsumptionPeak("evening_peak",     2,  datetime(2024, 1, 1, 19, 0),    datetime(2024, 1, 1, 19, 30)),
        ConsumptionPeak("Lunch",            1.5,datetime(2024, 1, 1, 13, 0),    datetime(2024, 1, 1, 13, 30)),
        ConsumptionPeak("Washing machine",  1.5,datetime(2024, 1, 1, 10, 30),   datetime(2024, 1, 1, 11, 0))
    ]

    # Solar production setup
    solar_peaks = [
        SolarProductionPeak(1.4,  datetime(2024, 1, 1, 9, 0),  datetime(2024, 1, 1, 11, 30)),
        SolarProductionPeak(2,    datetime(2024, 1, 1, 11, 0), datetime(2024, 1, 1, 17, 0))
    ]



    ## Data generation
    scenario = Scenario(ScenarioInputs(timestamps, BaseConsumption(mean_base=0.5, peaks=consumption_peaks), SolarProduction(peaks=solar_peaks)))

    

    ## Strategies configuration
    strategies = [
        ScheduledStrategy(
            "Night scheduling", [
            {"start": datetime(2024, 1, 1, 0, 0), "end": datetime(2024, 1, 1, 2, 0)},
            {"start": datetime(2024, 1, 1, 22, 0),"end": datetime(2024, 1, 2, 0, 0)}
        ]),
        ScheduledStrategy(
            "Day scheduling", [
            {"start": datetime(2024, 1, 1, 11, 0),
             "end": datetime(2024, 1, 1, 15, 0)}
        ]),
        SolarOnlyStrategy("100% Solar", threshold_start=1.1),
        MaximizeSolarStrategy("Max. Solar."),
        OptimizationStrategy("Optimiz", threshold=0.5)
    ]


    ## Strategies evalutation of CET optimization
    # CET properties
    cet_properties = CETProperties(power=1, min_duration=timedelta(minutes=15), max_duration=timedelta(hours=4))

    # Evaluate strategies and store results
    results = {}
    for idx, strategy in enumerate(strategies):
        # Run optimization for each strategy
        results[strategy.name] = strategy.run_optimization(scenario, cet_properties)



    ## Display results
    # Create figure for plotting
    strategies_to_display = ["Night scheduling", "100% Solar", "Max. Solar.","Optimiz"]
    metrics_to_display = ['import_reseau','export_reseau', 'taux_autoconsommation','cout_total','cout_moyen_kwh', 'cout_fonctionnement_cet','cet_solar_share']

    SolarOptimizationVisualizer.plot_all_results(scenario, results, metrics_to_display, strategies_to_display)

    plt.show()

if __name__ == "__main__":
    main()
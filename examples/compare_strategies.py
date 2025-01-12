from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from solar_optimization.core import *
from solar_optimization.devices.cet import CETProperties
from solar_optimization.strategies import *
from solar_optimization.visualization.plotter import *


def main():
    ## Configurations
    # Timestamps
    time_resolution = 5 #minutes
    timestamps = TimeSeriesConfig.create_timestamps(time_delta=timedelta(minutes=time_resolution))

    ## Data generation
    scenario = Scenario(ScenarioInputs(timestamps, 
                                   DefaultBaseConsumption.generate(DefaultConsumptionScenario.WEEKEND_DAY), 
                                   DefaultSolarProduction.generate(DefaultProductionScenario.SUMMER_SUNNY_ALL_DAY)))

    ScenarioDataVisualiser.plot_scenario(scenario)
    
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
        MaximizeSolarStrategy("Max. Solar"),
        OptimizationStrategy("Optimiz", threshold=0.5)
    ]


    ## Strategies evalutation of CET optimization
    # CET properties
    cet_properties = CETProperties(power=1, min_duration=timedelta(minutes=15), max_duration=timedelta(hours=4))

    # Evaluate strategies and store results
    results = {}
    for strategy in strategies:
        # Run optimization for each strategy
        results[strategy.name] = strategy.run_optimization(scenario, cet_properties)



    ## Display results
    # Create figure for plotting
    strategies_to_display = ["Night scheduling","Day scheduling", "100% Solar", "Max. Solar", "Optimiz"]
    metrics_to_display = ['import_reseau','export_reseau', 'production_totale','taux_autoconsommation','cout_total','cout_moyen_kwh', 'cout_fonctionnement_cet','cet_solar_share']

    SolarOptimizationVisualizer.plot_all_results(scenario, results, metrics_to_display, strategies_to_display)
    
    plt.show()

if __name__ == "__main__":
    main()
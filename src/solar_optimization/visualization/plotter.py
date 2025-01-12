from dataclasses import asdict, fields
import matplotlib.pyplot as plt

from solar_optimization.strategies.base import OptimizationResult
from solar_optimization.metrics.models import OptimizationMetrics
from solar_optimization.core.scenarios import Scenario

class SolarOptimizationVisualizer:
    @staticmethod
    def plot_strategy_result(scenario:Scenario, strategy_name, strategy_result:OptimizationResult,
                            ax: plt.Axes) -> None:
        """Plot results for a single strategy"""

        cet_consumption=strategy_result.cet_consumption
        metrics=strategy_result.metrics
        
        home_consumption = scenario.consumption_data + cet_consumption
        grid_exchanges = home_consumption - scenario.production_data
        
        time_delta = scenario.timestamps[1] - scenario.timestamps[0]
        dt_hours = (time_delta).total_seconds() / 3600
        width = dt_hours / (24)
        
        ax.bar(scenario.timestamps, home_consumption, width, label='Consommation', color='#FF6B6B', alpha=0.7)
        ax.bar(scenario.timestamps, -scenario.production_data, width, label='Production solaire', color='#4ECB71', alpha=0.7)
        ax.plot(scenario.timestamps, grid_exchanges, 'b-', label='Échanges réseau', linewidth=2)
        ax.plot(scenario.timestamps, metrics.cet_active, 'r-', label='CET actif', linewidth=2)
        
        ax.set_xlabel('Heure')
        ax.set_ylabel('Puissance (kW)')
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_title(f'Stratégie: {strategy_name}')
        #ax.set_ylim(-3, 3)


    @staticmethod
    def plot_comparison_table(results:dict[str, OptimizationResult], metrics_to_display: list[str],
                            ax_table: plt.Axes, strategies_to_compare: list[str]=[]) -> None:
        """Plot results for a single strategy"""
        
        table_data = []
        table_data.append(['Metriques'])
        
        all_metrics = [field.name for field in fields(OptimizationMetrics)]
        metrics_found = [metric for metric in metrics_to_display if metric in all_metrics]
        
        all_strategies = results.keys()

        if not strategies_to_compare:
            strategies_found = all_strategies
        else:
            strategies_found = [strategy for strategy in strategies_to_compare if strategy in all_strategies]

        # Create header column
        for metric in metrics_to_display:
            table_data.append([metric])

        # Add data columns
        for key, result in results.items():
            if key in strategies_found:
                # Add Strategy name
                table_data[0].append(f"{key}")
                
                result_dict = asdict(result.metrics)
                for indx_metric, metric in enumerate(metrics_found):
                    table_data[indx_metric+1].append(f"{result_dict[metric]:2f}")
        
        table = ax_table.table(cellText=table_data, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)


    @staticmethod
    def plot_all_results(scenario:Scenario, results:dict[str, OptimizationResult], metrics_to_display:list[str], strategies_to_display: list[str]=[]) -> None:
        
        if not strategies_to_display:
            strategies_to_display = results.keys()

        rows = (len(strategies_to_display) + 1)*3
        fig = plt.figure(figsize=(15, rows))
        gs = plt.GridSpec(len(strategies_to_display)+1, 1, hspace=1)
        idx = 0
        for name, result in results.items():
            if name in strategies_to_display:
                # Plot results for this strategy
                ax = fig.add_subplot(gs[idx])
                SolarOptimizationVisualizer.plot_strategy_result(scenario, name, result, ax)
                idx+=1

        ax_table = fig.add_subplot(gs[idx])
        ax_table.axis('tight')
        ax_table.axis('off')

        SolarOptimizationVisualizer.plot_comparison_table(results, metrics_to_display, ax_table, strategies_to_display)


class ScenarioDataVisualiser:
    @staticmethod
    def plot_scenario(scenario:Scenario):

        home_consumption = scenario.consumption_data
        grid_exchanges = home_consumption - scenario.production_data
        
        time_delta = scenario.timestamps[1] - scenario.timestamps[0]
        width = time_delta.total_seconds() / (3600 * 24)
        
        fig = plt.figure(figsize=(10, 3))
        ax = plt.axes()
        ax.bar(scenario.timestamps, home_consumption, width, label='Consommation', color='#FF6B6B', alpha=0.7)
        ax.bar(scenario.timestamps, -scenario.production_data, width, label='Production solaire', color='#4ECB71', alpha=0.7)
        ax.plot(scenario.timestamps, grid_exchanges, 'b-', label='Échanges réseau', linewidth=2)
        ax
        ax.set_xlabel('Heure')
        ax.set_ylabel('Puissance (kW)')
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_title('Scenario')
        ax.set_ylim(-3, 3)
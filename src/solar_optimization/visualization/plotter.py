from dataclasses import asdict, fields
import matplotlib.pyplot as plt
from enum import Enum
from numpy import cumsum


from solar_optimization.strategies.base import OptimizationResult
from solar_optimization.metrics.models import OptimizationMetrics
from solar_optimization.core.scenarios import Scenario

class VisualizationType(Enum):
    RAW = 1
    DELTA =2
    COST = 3

class SolarOptimizationVisualizer:
    @staticmethod
    def plot_strategy_result(scenario:Scenario, strategy_name, strategy_result:OptimizationResult,
                            ax: plt.Axes, display_type:VisualizationType = VisualizationType.RAW) -> None:
        """Plot results for a single strategy"""

        # TO DISPLAY PRODUCTION, CONSUMPTION & GRID EXCHANGE DATA
        if display_type == VisualizationType.RAW:
            cet_consumption=strategy_result.cet_consumption
            metrics=strategy_result.metrics
            time_delta = scenario.timestamps[1] - scenario.timestamps[0]
            dt_hours = (time_delta).total_seconds() / 3600
            
            home_consumption = (scenario.consumption_data + cet_consumption)*dt_hours
            home_production = (scenario.production_data)*dt_hours
            grid_exchanges = (home_consumption - home_production)
            
            width = dt_hours / (2*24)
            
            ax.bar(scenario.timestamps, home_consumption, width, label='Consommation', color='#FF6B6B', alpha=0.7)
            ax.bar(scenario.timestamps, -home_production, width, label='Production solaire', color='#4ECB71', alpha=0.7)
            ax.plot(scenario.timestamps, grid_exchanges, label='Échanges réseau', linewidth=2)
            mini1, maxi1= ax.get_ylim()
            bound1 = 1.1*max(abs(mini1), abs(maxi1))
            ax.set_ylim(-bound1, bound1)
            
            ax.set_xlabel('Heure')
            ax.set_ylabel('Energie (kWh)')
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
            plt.setp(ax.get_xticklabels(), rotation=45)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
            ax.legend(loc='upper right')
            ax.set_title(f'Stratégie: {strategy_name}')
            
            ax2 = ax.twinx()
            ax2.plot(scenario.timestamps, metrics.cet_is_active, 'r-', linewidth=2)
            ax2.set_ylabel('CET active')
            mini2, maxi2= ax2.get_ylim()
            bound2 = 1.1*max(abs(mini2), abs(maxi2))
            ax2.set_ylim(-bound2, bound2)

        elif display_type == VisualizationType.DELTA:
            cet_consumption=strategy_result.cet_consumption
            metrics=strategy_result.metrics
            time_delta = scenario.timestamps[1] - scenario.timestamps[0]
            dt_hours = (time_delta).total_seconds() / 3600
            
            home_consumption = (scenario.consumption_data + cet_consumption)*dt_hours
            home_production = (scenario.production_data)*dt_hours
            grid_exchanges = (home_consumption - home_production)
            
            home_consumption_from_grid = home_consumption*metrics.from_grid_ratio
            home_consumption_from_solar_panels = home_consumption*(1.-metrics.from_grid_ratio)
            exports = -grid_exchanges
            exports[grid_exchanges>0] = 0
            
            width = dt_hours / (2*24)
            
            ax.bar(scenario.timestamps, home_consumption_from_solar_panels, width, label='From solar panels', color='#15B01A', alpha=0.7)
            ax.bar(scenario.timestamps, home_consumption_from_grid, width, bottom= home_consumption_from_solar_panels, label='From grid', color='#0000FF', alpha=0.7)
            ax.bar(scenario.timestamps, -exports, width, label='Solar exports', color='#FFFF00', alpha=0.7)
            mini1, maxi1= ax.get_ylim()
            bound1 = 1.1*max(abs(mini1), abs(maxi1))
            ax.set_ylim(-bound1, bound1)
            
            ax.set_xlabel('Heure')
            ax.set_ylabel('Energy (kWh)')
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
            plt.setp(ax.get_xticklabels(), rotation=45)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
            ax.legend(loc='upper right')
            ax.set_title(f'Stratégie: {strategy_name}')
            

            ax2 = ax.twinx()
            ax2.plot(scenario.timestamps, metrics.cost, 'k-', label='€', linewidth=2)
            ax2.set_ylabel('Daily cumulated €')
            mini2, maxi2= ax2.get_ylim()
            bound2 = 1.1*max(abs(mini2), abs(maxi2))
            ax2.set_ylim(-bound2, bound2)
        
        elif display_type == VisualizationType.COST:
            # TODO: ADD RAW CONSUMPTION, PRODUCTION AND GRID EXCHANGE
            cet_consumption=strategy_result.cet_consumption
            metrics=strategy_result.metrics
            time_delta = scenario.timestamps[1] - scenario.timestamps[0]
            dt_hours = (time_delta).total_seconds() / 3600

            home_consumption = (scenario.consumption_data + cet_consumption)*dt_hours
            home_production = (scenario.production_data)*dt_hours
            grid_exchanges = (home_consumption - home_production)

            home_consumption_from_grid = home_consumption*metrics.from_grid_ratio
            grid_cost_imports = home_consumption_from_grid*metrics.tarif_import
            exports = -grid_exchanges
            exports[grid_exchanges>0] = 0
            grid_export_revenues = exports*metrics.tarif_export
            
            width = dt_hours / (2*24)
            
            ax.bar(scenario.timestamps, grid_cost_imports, width, label='Grid import cost', color='#FF0000', alpha=0.7)
            ax.bar(scenario.timestamps, -grid_export_revenues, width, label='Grid export revenues', color='#00FF00', alpha=0.7)
            mini1, maxi1= ax.get_ylim()
            bound1 = 1.1*max(abs(mini1), abs(maxi1))
            ax.set_ylim(-bound1, bound1)
            ax.set_xlabel('Time')
            ax.set_ylabel('Instant €')
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
            plt.setp(ax.get_xticklabels(), rotation=45)
            
            ax.legend(loc='upper left')
            ax.set_title(f'Stratégie: {strategy_name}')

            ax2 = ax.twinx()
            ax2.plot(scenario.timestamps, metrics.cost, label='€', linewidth =2)
            ax2.set_ylabel('Daily Cumulated cost in €')

            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

            mini2, maxi2= ax2.get_ylim()
            bound2 = 1.1*max(abs(mini2), abs(maxi2))
            ax2.set_ylim(-bound2, bound2)
            

    
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
    def plot_all_results(scenario:Scenario, results:dict[str, OptimizationResult], metrics_to_display:list[str], strategies_to_display: list[str]=[], display_type:VisualizationType = VisualizationType.RAW) -> None:
        
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
                SolarOptimizationVisualizer.plot_strategy_result(scenario, name, result, ax, display_type)
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
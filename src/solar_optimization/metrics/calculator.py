import numpy as np

from solar_optimization.metrics.models import OptimizationMetrics
from solar_optimization.core.scenarios import Scenario

class MetricsCalculator:
    def __init__(self, import_tarif:float=0.25, export_tarif:float=0.1269):
        self.import_tariff = import_tarif  # €/kWh
        self.export_tariff = export_tarif  # €/kWh

    def run(self, scenario:Scenario, cet_consumption: np.ndarray) -> OptimizationMetrics:
        
        dt_hours = (scenario.timestamps[1]-scenario.timestamps[0]).total_seconds() / 3600
        
        # Calculate total consumption and production
        home_consumption = scenario.consumption_data + cet_consumption
        grid_exchanges = home_consumption - scenario.production_data
        total_solar_production = np.sum(scenario.production_data) * dt_hours
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
        
        cet_active = np.zeros_like(scenario.timestamps)
        cet_active[cet_consumption > 0] = 1
        
        cet_solar_ratio = np.sum((1-consumption_from_grid_ratio) * cet_consumption) / np.sum(cet_consumption) * 100 if np.sum(cet_consumption) > 0 else 0
        
        cet_runtime_hours = np.sum(cet_active) * dt_hours
        
        cumulated_cost = np.cumsum(grid_kwh_price*home_consumption*dt_hours - grid_exports*dt_hours*self.export_tariff)

        return OptimizationMetrics(
            total_production=total_solar_production,
            total_consumption=total_home_consumption,
            total_grid_imports=total_grid_imports,
            total_grid_exports=total_grid_exports,
            total_self_consumption=self_consumption_rate,
            total_cost=total_cost,
            tarif_import = self.import_tariff,
            tarif_export = self.export_tariff,
            total_import_cost=import_cost,
            total_revenue_export=export_revenue,
            mean_cost_per_kwh=mean_cost_per_kwh,
            from_grid_ratio = consumption_from_grid_ratio,
            cost = cumulated_cost,
            cet_runtime_hours=cet_runtime_hours,
            cet_total_cost=cet_total_cost,
            cet_is_active=cet_active,
            cet_solar_share=cet_solar_ratio
        )
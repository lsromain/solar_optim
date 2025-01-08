import numpy as np

from .models import OptimizationMetrics
from ..core.scenarios import Scenario

class MetricsCalculator:
    def __init__(self):
        self.import_tariff = 0.25  # €/kWh
        self.export_tariff = 0.1269   # €/kWh

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
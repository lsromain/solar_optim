from dataclasses import dataclass
import numpy as np

@dataclass
class OptimizationMetrics:
    total_production: float  # kWh
    total_consumption: float  # kWh
    total_grid_imports: float  # kWh
    total_grid_exports: float  # kWh
    total_self_consumption: float  # %
    total_cost: float  # €
    tarif_import: float # €
    tarif_export: float # €
    total_import_cost: float  # €
    total_revenue_export: float  # €
    mean_cost_per_kwh: float  # €/kWh
    from_grid_ratio: np.ndarray  # % float array
    cost: float # € float array
    cet_runtime_hours: float  # hours
    cet_total_cost: float  # €
    cet_is_active: np.ndarray  # binary array
    cet_solar_share: float  # %

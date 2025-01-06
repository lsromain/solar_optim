from dataclasses import dataclass
import numpy as np

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
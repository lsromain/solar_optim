import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import savgol_filter

def create_time_data():
    start_time = datetime(2024, 1, 1, 7, 0)
    end_time = datetime(2024, 1, 1, 19, 0)
    time_delta = timedelta(minutes=5)
    
    timestamps = []
    current_time = start_time
    while current_time <= end_time:
        timestamps.append(current_time)
        current_time += time_delta
        
    t = np.linspace(0, 12, len(timestamps))
    return timestamps, t, time_delta

def create_base_consumption(t):
    n_points = len(t)
    mean_base_conso = 1
    base_consumption = np.ones(n_points) * mean_base_conso
    
    # Morning peak (7h-9h)
    morning_mask = (t >= 0) & (t <= 2)
    morning_peak = np.zeros_like(t)
    morning_peak[morning_mask] = 2 * np.sin(np.pi * (t[morning_mask]) / 2)
    
    # Evening peak (17h-19h)
    evening_mask = (t >= 10) & (t <= 12)
    evening_peak = np.zeros_like(t)
    evening_peak[evening_mask] = 2 * np.sin(np.pi * (t[evening_mask] - 10) / 2)
    
    # Combine and smooth
    base_consumption = base_consumption + morning_peak + evening_peak
    base_consumption += np.random.normal(0, 0.05, n_points)
    base_consumption = savgol_filter(base_consumption, window_length=10, polyorder=3)
    
    return base_consumption

def create_solar_production(t):
    production_start = 2   # 9h
    production_peak = 6    # 13h
    production_end = 11    # 18h
    peak_production = 2.5  # kW
    
    solar_production = np.zeros_like(t)
    mask = (t >= production_start) & (t <= production_end)
    t_normalized = (t[mask] - production_start) / (production_end - production_start)
    solar_production[mask] = -peak_production * np.sin(np.pi * t_normalized) * np.exp(-((t[mask] - production_peak) ** 2) / 8)
    
    return solar_production

def strategy_scheduled(t):
    """Stratégie 1: Programmation fixe"""
    cet_start = 4  # 11h
    cet_duration = 4  # 4 heures
    cet_power = 1.0  # 1 kW
    
    cet_consumption = np.zeros_like(t)
    cet_mask = (t >= cet_start) & (t <= cet_start + cet_duration)
    cet_consumption[cet_mask] = cet_power
    
    return cet_consumption

def strategy_solar_only(t, solar_production, base_consumption):
    """Stratégie 2: 100% solaire"""
    cet_power = 1.0  # 1 kW
    min_duration = 3  # 15 minutes (3 points de 5 minutes)
    cet_consumption = np.zeros_like(t)
    
    # Calculer la puissance disponible
    available_power = -solar_production - base_consumption
    
    # Initialiser les compteurs
    running_count = 0
    stopped_count = 0
    is_running = False
    
    for i in range(len(t)):
        if is_running:
            if available_power[i] <= 0 and running_count >= min_duration:
                is_running = False
                stopped_count = 0
            elif running_count < min_duration:
                cet_consumption[i] = cet_power
                running_count += 1
            else:
                cet_consumption[i] = cet_power
                running_count += 1
        else:
            if available_power[i] >= cet_power * 1.1 and stopped_count >= min_duration:
                is_running = True
                running_count = 0
                cet_consumption[i] = cet_power
            else:
                stopped_count += 1
    
    return cet_consumption

def strategy_maximize_solar(t, solar_production):
    """Stratégie 3: Maximiser l'autoconsommation"""
    cet_power = 1.0
    min_duration = 3  # 15 minutes (3 points de 5 minutes)
    cet_consumption = np.zeros_like(t)
    
    running_count = 0
    stopped_count = 0
    is_running = False
    
    for i in range(len(t)):
        if is_running:
            if solar_production[i] >= 0 and running_count >= min_duration:
                is_running = False
                stopped_count = 0
            elif running_count < min_duration:
                cet_consumption[i] = cet_power
                running_count += 1
            else:
                cet_consumption[i] = cet_power
                running_count += 1
        else:
            if solar_production[i] < 0 and stopped_count >= min_duration:
                is_running = True
                running_count = 0
                cet_consumption[i] = cet_power
            else:
                stopped_count += 1
    
    return cet_consumption

def plot_results(timestamps, base_consumption, solar_production, cet_consumptions, time_delta):
    strategies = ['Programmé', '100% Solaire', 'Max. Autoconso.']
    fig, axs = plt.subplots(3, 1, figsize=(12, 15))
    
    for idx, (strategy_name, cet_consumption) in enumerate(zip(strategies, cet_consumptions)):
        ax = axs[idx]
        consumption = base_consumption + cet_consumption
        grid_exchanges = consumption + solar_production
        
        width = time_delta.total_seconds() / (3600 * 24)
        ax.bar(timestamps, consumption, width, label='Consommation', color='#FF6B6B', alpha=0.7)
        ax.bar(timestamps, solar_production, width, label='Production solaire', color='#4ECB71', alpha=0.7)
        ax.plot(timestamps, grid_exchanges, 'b-', label='Échanges réseau', linewidth=2)
        
        ax.set_xlabel('Heure')
        ax.set_ylabel('Puissance (kW)')
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.legend(loc='upper right')
        ax.set_title(f'Stratégie: {strategy_name}')
        ax.set_ylim(-3, 3)
    
    plt.tight_layout()
    return fig

def main():
    # Création des données temporelles
    timestamps, t, time_delta = create_time_data()
    
    # Création des données de base
    base_consumption = create_base_consumption(t)
    solar_production = create_solar_production(t)
    
    # Application des trois stratégies
    cet_scheduled = strategy_scheduled(t)
    cet_solar_only = strategy_solar_only(t, solar_production, base_consumption)
    cet_maximize = strategy_maximize_solar(t, solar_production)
    
    # Création du graphique comparatif
    fig = plot_results(
        timestamps,
        base_consumption,
        solar_production,
        [cet_scheduled, cet_solar_only, cet_maximize],
        time_delta
    )
    
    plt.show()

if __name__ == "__main__":
    main()
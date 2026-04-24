import os
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    'figure.facecolor': '#0e1117',
    'axes.facecolor': '#1a1a2e',
    'text.color': '#e0e0e0',
    'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#aaaaaa',
    'ytick.color': '#aaaaaa',
    'axes.edgecolor': '#333366',
    'grid.color': '#333366',
    'font.family': 'sans-serif',
    'font.size': 11,
    'figure.dpi': 150,
})

PALETTE = ['#667eea', '#764ba2', '#f5576c', '#4facfe', '#38ef7d', '#f093fb']

def convert_mercator_to_wgs84(x: float, y: float) -> tuple[float, float]:
    """Converts Web Mercator coordinates to standard WGS84 GPS coordinates."""
    lng = (x / 20037508.34) * 180.0
    lat_rad = (y / 20037508.34) * 180.0
    lat = 180.0 / math.pi * (2 * np.arctan(np.exp(lat_rad * math.pi / 180.0)) - math.pi / 2)
    return lng, lat

def calculate_haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Calculates vectorized haversine distance between arrays of coordinates."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1
    a = np.sin(delta_lat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon/2)**2
    return 2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def load_delivery_data(file_path: str = "data/raw/lade/delivery_five_cities.csv") -> pd.DataFrame:
    """Loads, cleans, and engineers initial spatial-temporal features for the dataset."""
    print(f"Loading data from {file_path}")
    delivery_data = pd.read_csv(file_path)
    
    delivery_data['receipt_time'] = pd.to_datetime('2022-' + delivery_data['receipt_time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    delivery_data['sign_time'] = pd.to_datetime('2022-' + delivery_data['sign_time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    delivery_data = delivery_data.dropna(subset=['receipt_time', 'sign_time'])
    
    delivery_data['duration_mins'] = (delivery_data['sign_time'] - delivery_data['receipt_time']).dt.total_seconds() / 60.0
    delivery_data = delivery_data[(delivery_data['duration_mins'] > 1) & (delivery_data['duration_mins'] < 300)]
    
    delivery_data['receipt_wgs_lng'], delivery_data['receipt_wgs_lat'] = convert_mercator_to_wgs84(delivery_data['receipt_lng'], delivery_data['receipt_lat'])
    delivery_data['poi_wgs_lng'], delivery_data['poi_wgs_lat'] = convert_mercator_to_wgs84(delivery_data['poi_lng'], delivery_data['poi_lat'])
    delivery_data['distance_km'] = calculate_haversine_distance(delivery_data['receipt_wgs_lng'], delivery_data['receipt_wgs_lat'], delivery_data['poi_wgs_lng'], delivery_data['poi_wgs_lat'])
    delivery_data = delivery_data[(delivery_data['distance_km'] > 0.01) & (delivery_data['distance_km'] < 50)]
    
    delivery_data['hour'] = delivery_data['receipt_time'].dt.hour
    delivery_data['dow'] = delivery_data['ds'] % 7
    delivery_data['city'] = delivery_data['from_city_name'].fillna('Unknown')
    
    city_mapping = {city_name: f"City_{chr(65+i)}" for i, city_name in enumerate(delivery_data['city'].unique())}
    delivery_data['city_label'] = delivery_data['city'].map(city_mapping)
    
    courier_daily_loads = delivery_data.groupby(['delivery_user_id', 'ds']).size().reset_index(name='daily_packages')
    delivery_data = delivery_data.merge(courier_daily_loads, on=['delivery_user_id', 'ds'], how='left')
    
    print(f"Successfully loaded {len(delivery_data):,} verified deliveries.")
    return delivery_data

def plot_duration_distribution(delivery_data: pd.DataFrame, output_directory: str):
    """Generates distribution plots for delivery durations and distances."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(delivery_data['duration_mins'], bins=80, color=PALETTE[0], alpha=0.85, edgecolor='none')
    axes[0].axvline(delivery_data['duration_mins'].median(), color=PALETTE[2], linestyle='--', linewidth=2, label=f"Median: {delivery_data['duration_mins'].median():.0f} min")
    axes[0].axvline(delivery_data['duration_mins'].mean(), color=PALETTE[3], linestyle='--', linewidth=2, label=f"Mean: {delivery_data['duration_mins'].mean():.0f} min")
    axes[0].set_xlabel('Delivery Duration (minutes)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Delivery Durations')
    axes[0].legend(fontsize=9)
    axes[0].set_xlim(0, 300)
    
    axes[1].hist(delivery_data['distance_km'], bins=80, color=PALETTE[1], alpha=0.85, edgecolor='none')
    axes[1].axvline(delivery_data['distance_km'].median(), color=PALETTE[2], linestyle='--', linewidth=2, label=f"Median: {delivery_data['distance_km'].median():.2f} km")
    axes[1].set_xlabel('Delivery Distance (km)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of Delivery Distances')
    axes[1].legend(fontsize=9)
    axes[1].set_xlim(0, 10)
    
    plt.tight_layout()
    plt.savefig(f"{output_directory}/01_distributions.png", bbox_inches='tight')
    plt.close()

def plot_hourly_patterns(delivery_data: pd.DataFrame, output_directory: str):
    """Plots delivery volume and average duration sliced by hour of day."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    hourly_stats = delivery_data.groupby('hour').agg(
        count=('duration_mins', 'size'),
        avg_duration=('duration_mins', 'mean')
    ).reset_index()
    
    bars = axes[0].bar(hourly_stats['hour'], hourly_stats['count'], color=PALETTE[0], alpha=0.85)
    rush_hours = [8, 9, 10, 17, 18, 19]
    for bar, hour in zip(bars, hourly_stats['hour']):
        if hour in rush_hours:
            bar.set_color(PALETTE[2])
            
    axes[0].set_xlabel('Hour of Day')
    axes[0].set_ylabel('Number of Deliveries')
    axes[0].set_title('Delivery Volume by Hour')
    axes[0].set_xticks(range(0, 24, 2))
    
    axes[1].plot(hourly_stats['hour'], hourly_stats['avg_duration'], color=PALETTE[3], linewidth=2.5, marker='o', markersize=5)
    axes[1].fill_between(hourly_stats['hour'], hourly_stats['avg_duration'], alpha=0.15, color=PALETTE[3])
    for hour in rush_hours:
        axes[1].axvspan(hour-0.5, hour+0.5, alpha=0.08, color=PALETTE[2])
        
    axes[1].set_xlabel('Hour of Day')
    axes[1].set_ylabel('Avg Duration (minutes)')
    axes[1].set_title('Average Delivery Duration by Hour')
    axes[1].set_xticks(range(0, 24, 2))
    
    plt.tight_layout()
    plt.savefig(f"{output_directory}/02_hourly_patterns.png", bbox_inches='tight')
    plt.close()

def plot_city_comparison(delivery_data: pd.DataFrame, output_directory: str):
    """Generates comparative bar charts for different cities."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    city_stats = delivery_data.groupby('city_label').agg(
        count=('duration_mins', 'size'),
        avg_duration=('duration_mins', 'mean'),
        avg_distance=('distance_km', 'mean')
    ).reset_index().sort_values('count', ascending=False)
    
    axes[0].barh(city_stats['city_label'], city_stats['count'], color=PALETTE[:len(city_stats)])
    axes[0].set_xlabel('Number of Deliveries')
    axes[0].set_title('Delivery Volume by City')
    
    axes[1].barh(city_stats['city_label'], city_stats['avg_duration'], color=PALETTE[:len(city_stats)])
    axes[1].set_xlabel('Avg Duration (min)')
    axes[1].set_title('Avg Delivery Duration by City')
    
    axes[2].barh(city_stats['city_label'], city_stats['avg_distance'], color=PALETTE[:len(city_stats)])
    axes[2].set_xlabel('Avg Distance (km)')
    axes[2].set_title('Avg Delivery Distance by City')
    
    plt.tight_layout()
    plt.savefig(f"{output_directory}/03_city_comparison.png", bbox_inches='tight')
    plt.close()

def plot_distance_vs_duration(delivery_data: pd.DataFrame, output_directory: str):
    """Plots a scatter visualization of distance vs duration."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sample_data = delivery_data.sample(min(5000, len(delivery_data)), random_state=42)
    scatter_plot = ax.scatter(
        sample_data['distance_km'], sample_data['duration_mins'],
        c=sample_data['hour'], cmap='plasma', alpha=0.3, s=8, edgecolors='none'
    )
    plt.colorbar(scatter_plot, ax=ax, label='Hour of Day')
    
    poly_fit = np.polyfit(sample_data['distance_km'], sample_data['duration_mins'], 1)
    poly_function = np.poly1d(poly_fit)
    x_axis_line = np.linspace(0, 10, 100)
    ax.plot(x_axis_line, poly_function(x_axis_line), color=PALETTE[2], linewidth=2, linestyle='--', label=f'Trend: {poly_fit[0]:.1f} min/km')
    
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Duration (minutes)')
    ax.set_title('Distance vs Duration')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 250)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_directory}/04_distance_vs_duration.png", bbox_inches='tight')
    plt.close()

def plot_heatmap(delivery_data: pd.DataFrame, output_directory: str):
    """Generates a day-of-week by hour heatmap for delivery durations."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    pivot_data = delivery_data.pivot_table(values='duration_mins', index='dow', columns='hour', aggfunc='mean')
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    pivot_data.index = [day_labels[i] if i < len(day_labels) else str(i) for i in pivot_data.index]
    
    sns.heatmap(pivot_data, cmap='magma', ax=ax, linewidths=0.5, 
                cbar_kws={'label': 'Avg Duration (min)'}, annot=False)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Day of Week')
    ax.set_title('Average Delivery Duration: Day x Hour Heatmap')
    
    plt.tight_layout()
    plt.savefig(f"{output_directory}/05_heatmap_day_hour.png", bbox_inches='tight')
    plt.close()

def plot_courier_workload(delivery_data: pd.DataFrame, output_directory: str):
    """Analyzes and plots metrics regarding individual courier workloads."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    courier_stats = delivery_data.groupby('delivery_user_id').agg(
        total_packages=('duration_mins', 'size'),
        avg_duration=('duration_mins', 'mean'),
        avg_daily=('daily_packages', 'mean')
    ).reset_index()
    
    axes[0].hist(courier_stats['avg_daily'], bins=50, color=PALETTE[4], alpha=0.85, edgecolor='none')
    axes[0].set_xlabel('Avg Daily Packages per Courier')
    axes[0].set_ylabel('Number of Couriers')
    axes[0].set_title('Courier Workload Distribution')
    axes[0].axvline(courier_stats['avg_daily'].median(), color=PALETTE[2], linestyle='--', 
                     label=f"Median: {courier_stats['avg_daily'].median():.0f}")
    axes[0].legend()
    
    sample_couriers = courier_stats[courier_stats['total_packages'] > 50].sample(min(500, len(courier_stats)), random_state=42)
    axes[1].scatter(sample_couriers['avg_daily'], sample_couriers['avg_duration'],
                    alpha=0.4, s=15, color=PALETTE[5], edgecolors='none')
    axes[1].set_xlabel('Avg Daily Packages')
    axes[1].set_ylabel('Avg Delivery Duration (min)')
    axes[1].set_title('Workload vs Performance')
    
    plt.tight_layout()
    plt.savefig(f"{output_directory}/06_courier_workload.png", bbox_inches='tight')
    plt.close()

def generate_summary_stats(delivery_data: pd.DataFrame, output_directory: str):
    """Extracts and saves high-level descriptive statistics to a text file."""
    stats_dict = {
        "total_deliveries": len(delivery_data),
        "unique_couriers": delivery_data['delivery_user_id'].nunique(),
        "cities": delivery_data['city_label'].nunique(),
        "date_range_days": int(delivery_data['ds'].max() - delivery_data['ds'].min()),
        "avg_duration_mins": round(delivery_data['duration_mins'].mean(), 1),
        "median_duration_mins": round(delivery_data['duration_mins'].median(), 1),
        "avg_distance_km": round(delivery_data['distance_km'].mean(), 3),
        "median_distance_km": round(delivery_data['distance_km'].median(), 3),
        "rush_hour_pct": round((delivery_data['hour'].isin([8,9,10,17,18,19]).sum() / len(delivery_data)) * 100, 1),
    }
    
    with open(f"{output_directory}/dataset_stats.txt", 'w') as file:
        file.write("=" * 50 + "\n")
        file.write("  LaDe-D Dataset Summary Statistics\n")
        file.write("=" * 50 + "\n\n")
        for key, value in stats_dict.items():
            file.write(f"  {key:30s}: {value}\n")

if __name__ == "__main__":
    output_dir = "ml/eda_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    dataset = load_delivery_data()
    
    plot_duration_distribution(dataset, output_dir)
    plot_hourly_patterns(dataset, output_dir)
    plot_city_comparison(dataset, output_dir)
    plot_distance_vs_duration(dataset, output_dir)
    plot_heatmap(dataset, output_dir)
    plot_courier_workload(dataset, output_dir)
    generate_summary_stats(dataset, output_dir)

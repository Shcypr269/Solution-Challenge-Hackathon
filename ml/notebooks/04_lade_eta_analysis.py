import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians, cos, sin, asin, sqrt
import os

plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# Try to load LaDe delivery dataset
lade_path = '../data/raw/lade/delivery_five_cities.csv'

if not os.path.exists(lade_path):
    print("LaDe dataset not found")

    np.random.seed(42)
    n = 10000
    cities = ['shanghai', 'hangzhou', 'chongqing', 'jilin', 'yantai']
    
    df = pd.DataFrame({
        'package_id': range(n),
        'courier_id': np.random.randint(1, 500, n),
        'city': np.random.choice(cities, n),
        'lng': np.random.uniform(120.0, 122.0, n),
        'lat': np.random.uniform(30.0, 32.0, n),
        'accept_gps_lng': np.random.uniform(120.0, 122.0, n),
        'accept_gps_lat': np.random.uniform(30.0, 32.0, n),
        'delivery_gps_lng': np.random.uniform(120.0, 122.0, n),
        'delivery_gps_lat': np.random.uniform(30.0, 32.0, n),
        'accept_time': pd.date_range('2023-03-01', periods=n, freq='3min'),
        'ds': np.random.choice(range(301, 330), n),
    })
    # Simulate delivery time as accept_time + random duration
    df['delivery_time'] = df['accept_time'] + pd.to_timedelta(np.random.randint(5, 120, n), unit='m')
else:
    print("Loading LaDe dataset...")
    df = pd.read_csv(lade_path, nrows=100000)  # Load 100K rows for analysis

print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(df.head())

# Feature Engineering from LaDe

def haversine(lon1, lat1, lon2, lat2):
    """Calculate distance between two GPS points in km."""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * 6371 * asin(sqrt(a))

# Compute distance between accept and delivery GPS if columns exist
if 'accept_gps_lng' in df.columns and 'delivery_gps_lng' in df.columns:
    df['delivery_distance_km'] = df.apply(
        lambda r: haversine(
            r['accept_gps_lng'], r['accept_gps_lat'],
            r['delivery_gps_lng'], r['delivery_gps_lat']
        ), axis=1
    )
    
    plt.figure(figsize=(10, 5))
    sns.histplot(df['delivery_distance_km'].clip(0, 30), bins=50, kde=True)
    plt.title('Distribution of Delivery Distances (km)')
    plt.xlabel('Distance (km)')
    plt.show()

# Compute delivery duration if timestamps exist
if 'accept_time' in df.columns and 'delivery_time' in df.columns:
    df['accept_time'] = pd.to_datetime(df['accept_time'], errors='coerce')
    df['delivery_time'] = pd.to_datetime(df['delivery_time'], errors='coerce')
    df['delivery_duration_mins'] = (df['delivery_time'] - df['accept_time']).dt.total_seconds() / 60
    df['delivery_duration_mins'] = df['delivery_duration_mins'].clip(0, 300)
    
    plt.figure(figsize=(10, 5))
    sns.histplot(df['delivery_duration_mins'].dropna(), bins=50, kde=True, color='coral')
    plt.title('Distribution of Delivery Duration (minutes)')
    plt.xlabel('Duration (min)')
    plt.show()
    
    # Hour of day analysis
    df['accept_hour'] = df['accept_time'].dt.hour
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x='accept_hour', y='delivery_duration_mins', palette='viridis')
    plt.title('Delivery Duration by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Duration (min)')
    plt.show()

# City-level analysis
if 'city' in df.columns:
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x='city', y='delivery_duration_mins', palette='magma', errorbar='sd')
    plt.title('Average Delivery Duration by City')
    plt.ylabel('Duration (min)')
    plt.show()

print("\n=== LaDe Feature Engineering Summary ===")
print("Extractable features for ETA prediction:")
print("  - delivery_distance_km (haversine from accept to delivery GPS)")
print("  - accept_hour (time-of-day effect)")
print("  - city (city-level patterns)")
print("  - courier_id (courier-specific speed patterns)")
print("  - day_of_week (from ds field)")
print("  - delivery_duration_mins (target variable for ETA model)")

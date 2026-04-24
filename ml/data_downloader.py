import os
import json
import requests
import pandas as pd
from typing import Optional

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'DataSet Download', 'data_gov_in')


# data.gov.in catalog search & download

DATA_GOV_RESOURCES = {
    "freight_road_rail": {
        "search": "freight road transport railways",
        "description": "Freight and Passenger Movement by Road Transport and Railways",
        "direct_urls": [
            "https://data.gov.in/resource/freight-and-passenger-movement-road-transport-and-railways",
        ]
    },
    "highway_traffic_volume": {
        "search": "national highway traffic volume",
        "description": "Traffic Volume Count on National Highways",
        "direct_urls": [
            "https://data.gov.in/resource/traffic-volume-count-national-highways",
        ]
    },
    "road_accidents": {
        "search": "road accidents india",
        "description": "Road Accidents in India (year-wise, state-wise)",
        "direct_urls": [
            "https://data.gov.in/resource/state-wise-road-accidents-persons-killed-and-injured",
        ]
    },
    "registered_vehicles": {
        "search": "registered motor vehicles state",
        "description": "State-wise Registered Motor Vehicles",
        "direct_urls": [
            "https://data.gov.in/resource/state-wise-number-registered-motor-vehicles",
        ]
    },
    "rainfall_district": {
        "search": "rainfall district india IMD",
        "description": "District-level Monthly Rainfall (IMD)",
        "direct_urls": []
    },
}


def search_data_gov(query: str, api_key: Optional[str] = None, limit: int = 5) -> list:
    """
    Search data.gov.in catalog for datasets matching query.
    
    Usage:
        results = search_data_gov("freight road transport", api_key="YOUR_KEY")
    """
    base_url = "https://api.data.gov.in/resource"
    params = {
        "api-key": api_key or os.environ.get("DATA_GOV_API_KEY", ""),
        "format": "json",
        "limit": limit,
    }
    
    # data.gov.in catalog search  
    catalog_url = f"https://data.gov.in/search/site/{query.replace(' ', '%20')}"
    print(f"Search URL: {catalog_url}")
    print(f"Go to this URL in your browser, find the dataset, and copy the resource ID.")
    print(f"Then use: download_data_gov_resource(resource_id, api_key)")
    
    return []


def download_data_gov_resource(resource_id: str, api_key: str, 
                                filename: str = None, limit: int = 10000) -> Optional[pd.DataFrame]:
    """
    Download a specific resource from data.gov.in API.
    
    Args:
        resource_id: The resource identifier from the dataset page URL
        api_key: Your data.gov.in API key
        filename: Optional filename to save as CSV
        limit: Number of records to fetch
    """
    url = f"https://api.data.gov.in/resource/{resource_id}"
    params = {
        "api-key": api_key,
        "format": "json",
        "limit": limit,
    }
    
    print(f"Fetching from data.gov.in: {url}")
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        records = data.get("records", [])
        if not records:
            print(f"No records found for resource {resource_id}")
            return None
        
        dataframe = pd.DataFrame(records)
        print(f"Downloaded {len(dataframe)} records, {len(dataframe.columns)} columns")
        print(f"Columns: {list(dataframe.columns)}")
        
        if filename:
            os.makedirs(DATA_DIR, exist_ok=True)
            filepath = os.path.join(DATA_DIR, filename)
            dataframe.to_csv(filepath, index=False)
            print(f"Saved to {filepath}")
        
        return dataframe
        
    except requests.exceptions.RequestException as e:
        print(f"Download failed: {e}")
        return None


# IMD Weather Data Scraper

def download_imd_rainfall_data():
    """
    Instructions for getting IMD rainfall data.
    
    IMD doesn't have a clean API, so here are the manual steps:
    1. Go to: https://mausam.imd.gov.in/imd_latest/contents/rainfall_information.php
    2. Select state -> download monthly/annual tables
    3. Save as CSV in DataSet Download/imd/
    
    Alternative: Use data.gov.in search "rainfall IMD"
    """
    print("IMD Weather Data — Manual Download Steps")
    print("""
    1. Go to: https://mausam.imd.gov.in
    2. Click 'Climate' -> 'Rainfall' -> 'District Rainfall'
    3. Select each state, download the monthly table
    4. Save to: DataSet Download/imd/
    
    For warnings:
    1. Click 'Warnings' -> 'District Level Warnings'
    2. Select state -> copy the table
    
    data.gov.in alternative:
    - Search: "rainfall district india IMD"
    - Search: "cyclone track india"
    - Search: "flood affected districts india"
    """)


# DataCo Preprocessor

def preprocess_dataco(csv_path: str = "DataSet Download/DataCoSupplyChainDataset.csv") -> pd.DataFrame:
    """
    Preprocess the DataCo Supply Chain dataset for ML training.
    
    The DataCo dataset has 180K rows and 53 columns with Late_delivery_risk labels.
    This function extracts and engineers features relevant to our supply chain model.
    """
    print(f"Loading DataCo dataset from {csv_path}...")
    dataframe = pd.read_csv(csv_path, encoding='latin1')
    print(f"Raw: {len(dataframe)} rows, {len(dataframe.columns)} columns")
    
    # Key features for delay prediction
    features = dataframe[[
        'Late_delivery_risk',           # TARGET
        'Days for shipping (real)',      # actual days
        'Days for shipment (scheduled)', # planned days
        'Delivery Status',              # Late/Advance/On-time/Canceled
        'Shipping Mode',                # Standard/Second/First/Same Day
        'Order Region',                 # geographic region
        'Category Name',               # product category
        'Order City',                   # city
        'Order Country',               # country
        'Customer Segment',            # Consumer/Corporate/Home Office
        'Latitude',                     # GPS lat
        'Longitude',                    # GPS long
        'Order Item Quantity',          # quantity
        'Sales',                        # order value
        'Order Profit Per Order',       # profit
        'Benefit per order',            # benefit
    ]].copy()
    
    # Engineer delay magnitude
    features['delay_days'] = features['Days for shipping (real)'] - features['Days for shipment (scheduled)']
    features['delay_ratio'] = features['delay_days'] / features['Days for shipment (scheduled)'].clip(lower=1)
    
    # Clean up
    features = features.dropna(subset=['Late_delivery_risk', 'Shipping Mode'])
    
    print(f"Processed: {len(features)} rows")
    print(f"Late delivery risk: {features['Late_delivery_risk'].mean():.1%}")
    print(f"Delivery Status distribution:")
    for status, count in features['Delivery Status'].value_counts().items():
        print(f"  {status}: {count:,}")
    
    # Save processed
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'dataco_processed.csv')
    features.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    
    return features


# FASTag Feature Engineer

def engineer_fastag_features(csv_path: str = "DataSet Download/FastagFraudDetection.csv") -> pd.DataFrame:
    """
    Extract logistics-relevant features from FASTag fraud detection data.
    
    Even though this is fraud-focused, it contains:
    - TollBoothID: proxy for highway checkpoint locations
    - Timestamp: time-series of vehicle movements
    - Vehicle_Speed: speed at toll crossings
    - Geographical_Location: location strings
    - Vehicle_Type: vehicle classification
    """
    print(f"Loading FASTag data from {csv_path}...")
    dataframe = pd.read_csv(csv_path)
    print(f"Rows: {len(dataframe)}, Columns: {list(dataframe.columns)}")
    
    # Parse timestamps
    dataframe['Timestamp'] = pd.to_datetime(dataframe['Timestamp'], errors='coerce')
    dataframe = dataframe.dropna(subset=['Timestamp'])
    
    # Time features
    dataframe['hour'] = dataframe['Timestamp'].dt.hour
    dataframe['day_of_week'] = dataframe['Timestamp'].dt.dayofweek
    dataframe['is_night'] = dataframe['hour'].apply(lambda h: 1 if h < 6 or h > 22 else 0)
    dataframe['is_rush_hour'] = dataframe['hour'].apply(lambda h: 1 if (8 <= h <= 10) or (17 <= h <= 19) else 0)
    
    # Speed anomaly
    avg_speed = dataframe['Vehicle_Speed'].mean()
    std_speed = dataframe['Vehicle_Speed'].std()
    dataframe['speed_z_score'] = (dataframe['Vehicle_Speed'] - avg_speed) / std_speed
    dataframe['is_speed_anomaly'] = (dataframe['speed_z_score'].abs() > 2).astype(int)
    
    # Toll booth traffic patterns
    toll_traffic = dataframe.groupby(['TollBoothID', 'hour']).size().reset_index(name='hourly_volume')
    dataframe = dataframe.merge(toll_traffic, on=['TollBoothID', 'hour'], how='left')
    
    print(f"\nEngineered features:")
    print(f"  Unique toll booths: {dataframe['TollBoothID'].nunique()}")
    print(f"  Speed anomalies: {dataframe['is_speed_anomaly'].sum()}")
    print(f"  Night crossings: {dataframe['is_night'].sum()}")
    print(f"  Rush hour crossings: {dataframe['is_rush_hour'].sum()}")
    
    # Save
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'fastag_engineered.csv')
    dataframe.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    
    return dataframe


# Upply Seaports India Filter

def extract_india_ports(csv_path: str = "DataSet Download/UPPLY-SEAPORTS.csv") -> pd.DataFrame:
    """Extract Indian port nodes from Upply global seaports data."""
    print(f"Loading Upply seaports from {csv_path}...")
    
    # This CSV uses semicolons
    dataframe = pd.read_csv(csv_path, sep=';', names=['code', 'name', 'latitude', 'longitude', 'country_code', 'zone_code'],
                     skiprows=1, on_bad_lines='skip')
    
    india_ports = dataframe[dataframe['country_code'] == 'IN'].copy()
    print(f"Total ports: {len(dataframe)}, India ports: {len(india_ports)}")
    
    if len(india_ports) > 0:
        print(f"\nTop Indian ports:")
        for _, row in india_ports.head(15).iterrows():
            print(f"  {row['code']}: {row['name']} ({row['latitude']}, {row['longitude']})")
    
    # Save
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'india_ports.csv')
    india_ports.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    
    return india_ports


# Main

if __name__ == '__main__':
    print("  Smart Supply Chain — Dataset Preparation Pipeline")
    
    # 1. Preprocess DataCo
    print("\sample_size--- [1/3] DataCo Supply Chain (180K rows) ---")
    try:
        dataco = preprocess_dataco()
    except Exception as e:
        print(f"  Skipped: {e}")
    
    # 2. Engineer FASTag features
    print("\sample_size--- [2/3] FASTag Toll Data ---")
    try:
        fastag = engineer_fastag_features()
    except Exception as e:
        print(f"  Skipped: {e}")
    
    # 3. Extract India ports
    print("\sample_size--- [3/3] India Port Nodes ---")
    try:
        ports = extract_india_ports()
    except Exception as e:
        print(f"  Skipped: {e}")
    
    # 4. Show what's still needed
    print("\sample_size" + "=" * 60)
    print("  REMAINING DATA GAPS (manual download required)")
    print("""
    [CRITICAL] Weather data:
      -> Go to data.gov.in, search: "rainfall district india IMD"
      -> Download CSV, save to: DataSet Download/data_gov_in/
    
    [RESOLVED VIA TOMTOM API] Highway traffic & Road accidents:
      -> Live traffic flow, congestion, and incident detection is now
      -> fully handled by the TomTom Traffic API integration.
      
    [IMPORTANT] Indian Railway Freight Data:
      -> Go to data.gov.in, search: "indian railways freight traffic"
      -> Needed for multi-modal routing accuracy.
    
    [IMPORTANT] ULIP FASTag movement:
      -> Register at goulip.in (2-3 day approval)
    
    [NICE-TO-HAVE] LEADS 2024:
      -> Download from dpiit.gov.in/logistics
    """)

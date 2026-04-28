import pandas as pd
from typing import Dict, Any

class FeatureEngineer:
    def __init__(self):
        pass
    def create_features(self, shipment_data: Dict[str, Any]) -> pd.DataFrame:
        features = {
            "delivery_partner": shipment_data.get("delivery_partner", "delhivery"),
            "package_type": shipment_data.get("package_type", "electronics"),
            "vehicle_type": shipment_data.get("vehicle_type", "truck"),
            "delivery_mode": shipment_data.get("delivery_mode", "standard"),
            "region": shipment_data.get("region", "central"),
            "weather_condition": shipment_data.get("weather_condition", "clear"),
            "distance_km": shipment_data.get("distance_km", 100.0),
            "package_weight_kg": shipment_data.get("package_weight_kg", 5.0)
        }
        
        if shipment_data.get("has_disruption", False):
            features["distance_km"] += 150.0 
            
        return pd.DataFrame([features])

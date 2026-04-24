from app.ml.delay_predictor import DelayPredictor
from app.ml.feature_engineering import FeatureEngineer
from typing import Dict, Any

predictor = DelayPredictor()
engineer = FeatureEngineer()

async def predict_shipment_delay(shipment_data: Dict[str, Any]) -> Dict[str, Any]:
    features = engineer.create_features(shipment_data)
    is_delayed, confidence = predictor.predict_delay_risk(features)
    
    return {
        "is_delayed": is_delayed,
        "confidence": float(confidence),
        "risk_level": "HIGH" if confidence > 0.7 else "MEDIUM" if confidence > 0.4 else "LOW"
    }

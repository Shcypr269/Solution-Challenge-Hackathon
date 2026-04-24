from app.agents.state import AgentState
from app.ml.delay_predictor import DelayPredictor
from app.ml.feature_engineering import FeatureEngineer

predictor = DelayPredictor()
engineer = FeatureEngineer()

async def shipment_agent_node(state: AgentState) -> dict:
    print("Agent 2: Predicting delay risk...")
    
    shipment_context = {
        "delivery_partner": "dhl",
        "package_type": "electronics",
        "vehicle_type": "ev van",
        "delivery_mode": "express",
        "region": "east",
        "weather_condition": "clear",
        "distance_km": 250.0,
        "package_weight_kg": 12.0
    }
    
    if state.get("disruption_details"):
        shipment_context["has_disruption"] = state.get("severity_level") in ["HIGH", "CRITICAL"]
    
    features = engineer.create_features(shipment_context)
    is_delayed, confidence = predictor.predict_delay_risk(features)
    
    return {
        "delay_prediction": {
            "is_delayed": is_delayed,
            "confidence": float(confidence)
        }
    }

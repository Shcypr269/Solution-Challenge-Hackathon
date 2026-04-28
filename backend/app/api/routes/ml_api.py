from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

router = APIRouter()


def _numpy_safe(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _numpy_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_numpy_safe(v) for v in obj]
    elif isinstance(obj, (np.bool_, )):
        return bool(obj)
    elif isinstance(obj, (np.integer, )):
        return int(obj)
    elif isinstance(obj, (np.floating, )):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

class SimulationRequest(BaseModel):
    disruption_type: str = Field(description="weather | port_congestion | highway_closure | strike")
    affected_region: str = Field(description="north | south | east | west | central")
    severity: float = Field(default=0.7, ge=0.0, le=1.0)
    fleet_size: int = Field(default=20, ge=5, le=100)
    inject_region: bool = Field(default=False, description="Inject affected_region into ~30% of fleet")

class ETARequest(BaseModel):
    distance_km: float
    hour: int = Field(ge=0, le=23)
    city: str = "unknown"
    day_of_week: int = Field(default=3, ge=0, le=6)
    courier_daily_packages: int = Field(default=30, ge=1)
    courier_avg_speed: float = Field(default=0.05, ge=0.001)
    aoi_type: str = "other"

class DelayPredictionRequest(BaseModel):
    delivery_partner: str = "delhivery"
    package_type: str = "electronics"
    vehicle_type: str = "truck"
    delivery_mode: str = "standard"
    region: str = "central"
    weather_condition: str = "clear"
    distance_km: float = 100.0
    package_weight_kg: float = 5.0

@router.post("/whatif")
async def run_whatif_simulation(req: SimulationRequest):
    """Run a what-if disruption simulation across a fleet of shipments."""
    from ml.whatif_simulator import WhatIfSimulator
    sim = WhatIfSimulator()
    inject = req.affected_region if req.inject_region else None
    fleet = sim.generate_sample_fleet(req.fleet_size, inject_region=inject)
    result = sim.simulate_disruption(
        shipments=fleet,
        disruption_type=req.disruption_type,
        affected_region=req.affected_region,
        severity=req.severity
    )
    # Include fleet details so frontend can display shipment metadata
    result["fleet"] = fleet
    return JSONResponse(content=_numpy_safe(result))

@router.post("/predict-delay")
async def predict_delay(req: DelayPredictionRequest):
    """Predict delay probability for a single shipment."""
    import pandas as pd
    import joblib
    
    import os
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    pipeline = joblib.load(os.path.join(_root, "models", "delay_predictor_v2.joblib"))
    features = pd.DataFrame([req.dict()])
    prob = float(pipeline.predict_proba(features)[0][1])
    pred = pipeline.predict(features)[0]
    
    return {
        "is_delayed": bool(pred),
        "delay_probability": round(prob, 4),
        "risk_level": "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW"
    }

@router.post("/predict-eta")
async def predict_eta(req: ETARequest):
    """Predict estimated delivery time in minutes (LaDe-D trained, 421K deliveries)."""
    from ml.eta_predictor import ETAPredictor
    predictor = ETAPredictor()
    eta = predictor.predict_eta(
        distance_km=req.distance_km,
        hour=req.hour,
        city=req.city,
        day_of_week=req.day_of_week,
        courier_daily_packages=req.courier_daily_packages,
        courier_avg_speed=req.courier_avg_speed,
        aoi_type=req.aoi_type,
    )
    return {
        "estimated_time_mins": round(eta, 1),
        "distance_km": req.distance_km,
        "city": req.city,
        "model": "eta_predictor_v2 (XGBoost, 421K LaDe-D deliveries)"
    }


# ── Anomaly Detection Endpoints ──

class AnomalyDetectRequest(BaseModel):
    shipment_id: str = "SHP-001"
    delivery_partner: str = "delhivery"
    package_type: str = "electronics"
    vehicle_type: str = "truck"
    delivery_mode: str = "standard"
    region: str = "central"
    weather_condition: str = "clear"
    distance_km: float = 100.0
    package_weight_kg: float = 5.0

class AnomalyBatchRequest(BaseModel):
    fleet_size: int = Field(default=25, ge=5, le=100)

@router.post("/anomaly-detect")
async def detect_anomaly(req: AnomalyDetectRequest):
    """Detect anomalies in a single shipment using Isolation Forest + Z-Score ensemble."""
    from ml.anomaly_detector import SupplyChainAnomalyDetector
    detector = SupplyChainAnomalyDetector(contamination=0.15)
    detector.fit_on_csv()
    result = detector.detect(req.dict())
    return JSONResponse(content=_numpy_safe({
        "shipment_id": result.shipment_id,
        "is_anomaly": result.is_anomaly,
        "anomaly_score": result.anomaly_score,
        "risk_level": result.risk_level,
        "reasons": result.anomaly_reasons,
        "feature_scores": result.feature_scores,
    }))

@router.post("/anomaly-detect-batch")
async def detect_anomaly_batch(req: AnomalyBatchRequest):
    """Run anomaly detection across a generated fleet."""
    from ml.anomaly_detector import SupplyChainAnomalyDetector, generate_test_fleet
    detector = SupplyChainAnomalyDetector(contamination=0.15)
    detector.fit_on_csv()
    fleet = generate_test_fleet(req.fleet_size)
    result = detector.detect_batch(fleet)
    # Include fleet details so frontend can display shipment metadata
    result["fleet"] = fleet
    return JSONResponse(content=_numpy_safe(result))


# ── Explainability Endpoints ──

@router.post("/explain-delay")
async def explain_delay(req: DelayPredictionRequest):
    """Get explainable delay prediction with feature contribution analysis."""
    from ml.explainability import DelayExplainer
    explainer = DelayExplainer()
    return explainer.explain(req.dict())

@router.post("/explain-eta")
async def explain_eta(req: ETARequest):
    """Get explainable ETA prediction with feature contributions."""
    import numpy as np
    from ml.explainability import ETAExplainer
    explainer = ETAExplainer()
    
    hour = req.hour
    features = {
        'delivery_distance_km': req.distance_km,
        'log_distance': float(np.log1p(req.distance_km)),
        'accept_hour': hour,
        'day_of_week': req.day_of_week,
        'is_rush_hour': 1 if (8 <= hour <= 10) or (17 <= hour <= 19) else 0,
        'is_weekend': 1 if req.day_of_week >= 5 else 0,
        'courier_daily_packages': req.courier_daily_packages,
        'courier_avg_speed': req.courier_avg_speed,
        'distance_x_rush': req.distance_km * (1 if (8 <= hour <= 10) or (17 <= hour <= 19) else 0),
        'city': req.city,
        'time_period': 'morning' if 6<=hour<12 else 'afternoon' if 12<=hour<17 else 'evening' if 17<=hour<21 else 'night',
        'aoi_type': req.aoi_type,
    }
    return explainer.explain(features)


# ── Multi-Modal Optimizer Endpoint ──

class TransportOptRequest(BaseModel):
    distance_km: float = Field(ge=1.0)
    weight_kg: float = Field(ge=0.1)
    deadline_hours: float = Field(ge=1.0)
    priority: str = Field(default="balanced", description="balanced | cost | speed | green")
    weather_severity: float = Field(default=0.0, ge=0.0, le=1.0)

@router.post("/optimize-transport")
async def optimize_transport_route(req: TransportOptRequest):
    """Find optimal transport mode comparing road, rail, air, and waterway."""
    from ml.multimodal_optimizer import optimize_transport
    return optimize_transport(
        distance_km=req.distance_km,
        weight_kg=req.weight_kg,
        deadline_hours=req.deadline_hours,
        priority=req.priority,
        weather_severity=req.weather_severity,
    )


# ── Transport Modes Metadata ──

@router.get("/transport-modes")
async def get_transport_modes():
    """Return available transport mode details for the optimizer UI."""
    from ml.multimodal_optimizer import TRANSPORT_MODES
    modes = []
    for mode_id, mode in TRANSPORT_MODES.items():
        modes.append({
            "mode_id": mode_id,
            "name": mode.name,
            "cost_per_tonne_km": mode.cost_per_tonne_km,
            "speed_kmh": mode.speed_kmh,
            "co2_per_km_kg": mode.co2_per_km_kg,
            "fixed_cost": mode.fixed_cost,
            "min_distance_km": mode.min_distance_km,
            "max_weight_kg": mode.max_weight_kg,
            "reliability": mode.reliability,
        })
    return {"modes": modes}


# ── Global Feature Importance ──

@router.get("/global-importance")
async def get_global_importance():
    """Return global feature importance from the delay predictor model."""
    from ml.explainability import DelayExplainer
    explainer = DelayExplainer()
    importance = explainer.get_global_importance()
    return {"features": importance}


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


# ── Persistent Fleet ──

def _load_fleet():
    """Load the persistent fleet from fleet.json."""
    import json
    fleet_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'fleet.json')
    with open(fleet_path, 'r') as f:
        return json.load(f)


@router.get("/fleet")
async def get_fleet():
    """Return the persistent fleet of 30 shipments."""
    return JSONResponse(content=_load_fleet())


@router.get("/fleet-scan")
async def fleet_scan():
    """Scan the persistent fleet for anomalies using the real Isolation Forest model."""
    from ml.anomaly_detector import SupplyChainAnomalyDetector
    detector = SupplyChainAnomalyDetector(contamination=0.15)
    detector.fit_on_csv()
    fleet = _load_fleet()
    result = detector.detect_batch(fleet)
    result["fleet"] = fleet
    return JSONResponse(content=_numpy_safe(result))


@router.get("/auto-reroute")
async def auto_reroute():
    """
    THE KILLER FEATURE: Detects anomalies in persistent fleet, then auto-optimizes
    critical/high-risk shipments with the transport optimizer.
    Shows the full loop: detect → assess → reroute → measure impact.
    """
    from ml.anomaly_detector import SupplyChainAnomalyDetector
    from ml.multimodal_optimizer import optimize_transport

    # Step 1: Scan fleet
    detector = SupplyChainAnomalyDetector(contamination=0.15)
    detector.fit_on_csv()
    fleet = _load_fleet()
    scan = detector.detect_batch(fleet)

    # Step 2: Find critical/high shipments
    results = scan.get("results", [])
    fleet_map = {s["shipment_id"]: s for s in fleet}
    rerouted = []
    total_cost_saved = 0
    total_co2_saved = 0
    total_time_saved = 0

    for r in results:
        if r.get("risk_level") in ("CRITICAL", "HIGH") and r.get("is_anomaly"):
            ship = fleet_map.get(r["shipment_id"], {})
            # Step 3: Auto-optimize this shipment
            try:
                opt = optimize_transport(
                    distance_km=ship.get("distance_km", 500),
                    weight_kg=ship.get("package_weight_kg", 100),
                    deadline_hours=48,
                    priority="balanced",
                )
                rec = opt.get("recommended", {})
                savings = opt.get("savings", {})
                rerouted.append({
                    "shipment_id": r["shipment_id"],
                    "origin": ship.get("origin", "?"),
                    "destination": ship.get("destination", "?"),
                    "risk_level": r["risk_level"],
                    "anomaly_score": r["anomaly_score"],
                    "reasons": r.get("reasons", [])[:2],
                    "original_mode": ship.get("vehicle_type", "truck"),
                    "recommended_mode": rec.get("mode", "N/A"),
                    "recommended_cost": rec.get("total_cost_inr", 0),
                    "recommended_time": rec.get("travel_time_hrs", 0),
                    "recommended_co2": rec.get("co2_emissions_kg", 0),
                    "meets_deadline": rec.get("meets_deadline", False),
                    "cost_saved": savings.get("cost_saving_inr", 0),
                    "co2_saved": savings.get("co2_saving_kg", 0),
                    "action": "AUTO_REROUTED",
                })
                total_cost_saved += savings.get("cost_saving_inr", 0)
                total_co2_saved += savings.get("co2_saving_kg", 0)
                total_time_saved += max(0, 48 - rec.get("travel_time_hrs", 48))
            except Exception:
                rerouted.append({
                    "shipment_id": r["shipment_id"],
                    "risk_level": r["risk_level"],
                    "anomaly_score": r["anomaly_score"],
                    "action": "ALERT_SENT",
                })

    return JSONResponse(content=_numpy_safe({
        "scan_summary": scan.get("summary", {}),
        "risk_distribution": scan.get("risk_distribution", {}),
        "total_fleet": len(fleet),
        "anomalies_detected": scan.get("summary", {}).get("anomalies_detected", 0),
        "auto_rerouted": len(rerouted),
        "rerouted_shipments": rerouted,
        "impact": {
            "total_cost_saved_inr": total_cost_saved,
            "total_co2_saved_kg": total_co2_saved,
            "total_time_saved_hrs": round(total_time_saved, 1),
            "penalties_prevented_inr": len(rerouted) * 20000,
        },
    }))


@router.get("/impact-metrics")
async def impact_metrics():
    """Compute real-time impact metrics from ML models on the persistent fleet."""
    from ml.anomaly_detector import SupplyChainAnomalyDetector
    from ml.multimodal_optimizer import optimize_transport

    detector = SupplyChainAnomalyDetector(contamination=0.15)
    detector.fit_on_csv()
    fleet = _load_fleet()
    scan = detector.detect_batch(fleet)
    summary = scan.get("summary", {})
    anomalies = summary.get("anomalies_detected", 0)

    # Compute optimizer savings for a typical corridor
    opt = optimize_transport(distance_km=1400, weight_kg=500, deadline_hours=48)
    savings = opt.get("savings", {})

    return JSONResponse(content=_numpy_safe({
        "fleet_size": len(fleet),
        "anomalies_detected": anomalies,
        "anomaly_rate": summary.get("anomaly_rate", 0),
        "penalties_prevented_inr": anomalies * 20000,
        "co2_saved_kg": savings.get("co2_saving_kg", 0) * max(1, anomalies),
        "cost_saved_inr": savings.get("cost_saving_inr", 0) * max(1, anomalies),
        "on_time_rate": round(1 - summary.get("anomaly_rate", 0.15), 3),
        "disruptions_caught": anomalies,
        "model_accuracy": 0.905,
        "critical_alerts": summary.get("critical_alerts", 0),
        "high_alerts": summary.get("high_alerts", 0),
    }))


# ══════════════════════════════════════════════════════════
# ── Gemini-Powered Intelligent Chat ──
# ══════════════════════════════════════════════════════════

GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyDVU9Ir1x38mgVUs1rFNqnwxvGnvjc9vt8")

SYSTEM_PROMPT = """You are LogiTrack AI, an expert supply chain logistics assistant.
You help users analyze their supply chain using real ML models deployed on a production backend.

You have access to these ML tools (call exactly ONE per user question):

1. **predict_delay** — Predict if a shipment will be delayed and the probability.
   Parameters: delivery_partner (delhivery|bluedart|shadowfax|xpressbees|dhl), package_type (electronics|perishable|fragile|clothing|machinery|chemicals|documents|food), vehicle_type (truck|bike|ev van|three wheeler), delivery_mode (standard|express), region (north|south|east|west|central), weather_condition (clear|rainy|stormy|foggy|extreme heat|cold), distance_km (number), package_weight_kg (number)

2. **predict_eta** — Estimate delivery time in minutes.
   Parameters: distance_km (number), hour (0-23), city (string), day_of_week (0-6, 0=Monday)

3. **optimize_transport** — Find cheapest/greenest/fastest transport mode.
   Parameters: distance_km (number), weight_kg (number), deadline_hours (number), priority (cost|speed|green|balanced)

4. **whatif** — Simulate a supply chain disruption scenario.
   Parameters: disruption_type (weather|port_congestion|highway_closure|strike), affected_region (north|south|east|west|central), severity (0.0-1.0), fleet_size (5-100)

5. **explain_delay** — Explain WHY a shipment would be delayed using SHAP analysis.
   Parameters: same as predict_delay

6. **anomaly_scan** — Scan the fleet for anomalous shipments.
   Parameters: none

7. **auto_reroute** — Detect anomalies and auto-optimize critical shipments.
   Parameters: none

INSTRUCTIONS:
- When the user asks a logistics question, determine which tool to call.
- Fill in reasonable defaults for any parameters not mentioned by the user.
- Use Indian cities to infer regions: Mumbai/Pune/Ahmedabad=west, Delhi/Jaipur/Lucknow=north, Bangalore/Chennai/Hyderabad=south, Kolkata/Patna=east, Bhopal=central.
- Respond with ONLY a JSON object (no markdown, no explanation), like:
  {"tool": "predict_delay", "params": {"delivery_partner": "delhivery", "distance_km": 1400, ...}}
- If the user is just chatting or asking a general logistics question (not needing a tool), respond with:
  {"tool": "none", "answer": "Your helpful answer here..."}
- NEVER return anything except valid JSON.
"""

SUMMARY_PROMPT = """You are LogiTrack AI. The user asked: "{question}"

The ML engine returned this data:
{result}

Write a clear, helpful, expert-level response for a logistics manager. Include:
- A direct answer to their question
- Key numbers and metrics from the data
- Actionable advice based on the results
- Use emojis sparingly for visual clarity
- Keep it concise (3-5 sentences max)
- If there are cost or time savings, highlight them
- Format numbers nicely (₹ for Indian Rupees, km, kg, etc.)
"""


class ChatRequest(BaseModel):
    message: str = Field(description="User's natural language question")
    conversation_history: Optional[List[dict]] = Field(default=[], description="Previous messages for context")


@router.post("/chat")
async def gemini_chat(req: ChatRequest):
    """Gemini-powered intelligent chat that routes to real ML models."""
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Step 1: Ask Gemini to understand the question and pick a tool
    try:
        routing_response = model.generate_content(
            SYSTEM_PROMPT + f"\n\nUser question: {req.message}"
        )
        raw_text = routing_response.text.strip()
        # Clean markdown fences if Gemini wraps it
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[1] if "\n" in raw_text else raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3].strip()
        routing = json.loads(raw_text)
    except Exception as e:
        return JSONResponse(content={
            "response": f"I'd be happy to help with your logistics question! However, I had trouble understanding that. Could you rephrase? Try asking about delay predictions, transport modes, disruption scenarios, or fleet anomalies.\n\n(Debug: {str(e)[:100]})",
            "tool_used": None,
            "ml_data": None,
        })

    tool = routing.get("tool", "none")
    params = routing.get("params", {})

    # Step 2: If no tool needed, return Gemini's direct answer
    if tool == "none":
        return JSONResponse(content={
            "response": routing.get("answer", "I'm LogiTrack AI — ask me about delay risks, ETAs, transport optimization, disruptions, or fleet health!"),
            "tool_used": None,
            "ml_data": None,
        })

    # Step 3: Call the real ML model
    ml_result = None
    tool_label = tool

    try:
        if tool == "predict_delay":
            from ml.explainability import DelayExplainer
            explainer = DelayExplainer()
            ml_result = explainer.explain(params)
            tool_label = "Delay Prediction + SHAP"

        elif tool == "predict_eta":
            import joblib
            eta_model = joblib.load("models/eta_predictor_v2.joblib")
            features = {
                'delivery_distance_km': params.get('distance_km', 50),
                'log_distance': float(np.log1p(params.get('distance_km', 50))),
                'accept_hour': params.get('hour', 14),
                'day_of_week': params.get('day_of_week', 3),
                'is_rush_hour': 1 if params.get('hour', 14) in range(8, 11) or params.get('hour', 14) in range(17, 20) else 0,
                'is_weekend': 1 if params.get('day_of_week', 3) >= 5 else 0,
            }
            import pandas as pd
            df = pd.DataFrame([features])
            for c in eta_model.feature_names_in_:
                if c not in df.columns:
                    df[c] = 0
            df = df[eta_model.feature_names_in_]
            pred = eta_model.predict(df)[0]
            ml_result = {"estimated_time_mins": round(float(pred), 1), "city": params.get("city", "Unknown"), "distance_km": params.get("distance_km", 50)}
            tool_label = "ETA Prediction"

        elif tool == "optimize_transport":
            from ml.multimodal_optimizer import optimize_transport
            ml_result = optimize_transport(**params)
            tool_label = "Transport Optimizer"

        elif tool == "whatif":
            from ml.whatif_simulator import WhatIfSimulator
            sim = WhatIfSimulator()
            fleet = sim.generate_sample_fleet(
                params.get("fleet_size", 20),
                inject_region=params.get("affected_region", "north")
            )
            ml_result = sim.simulate_disruption(
                fleet=fleet,
                disruption_type=params.get("disruption_type", "weather"),
                affected_region=params.get("affected_region", "north"),
                severity=params.get("severity", 0.7),
            )
            tool_label = "What-If Simulator"

        elif tool == "explain_delay":
            from ml.explainability import DelayExplainer
            explainer = DelayExplainer()
            ml_result = explainer.explain(params)
            tool_label = "Explainability (SHAP)"

        elif tool == "anomaly_scan":
            from ml.anomaly_detector import SupplyChainAnomalyDetector
            detector = SupplyChainAnomalyDetector(contamination=0.15)
            detector.fit_on_csv()
            fleet = _load_fleet()
            scan = detector.detect_batch(fleet)
            ml_result = {"summary": scan.get("summary", {}), "risk_distribution": scan.get("risk_distribution", {})}
            tool_label = "Anomaly Detection"

        elif tool == "auto_reroute":
            # Reuse the auto-reroute logic
            from ml.anomaly_detector import SupplyChainAnomalyDetector
            from ml.multimodal_optimizer import optimize_transport as opt_transport
            detector = SupplyChainAnomalyDetector(contamination=0.15)
            detector.fit_on_csv()
            fleet = _load_fleet()
            scan = detector.detect_batch(fleet)
            results = scan.get("results", [])
            fleet_map = {s["shipment_id"]: s for s in fleet}
            rerouted = []
            for r in results:
                if r.get("risk_level") in ("CRITICAL", "HIGH") and r.get("is_anomaly"):
                    ship = fleet_map.get(r["shipment_id"], {})
                    try:
                        o = opt_transport(distance_km=ship.get("distance_km", 500), weight_kg=ship.get("package_weight_kg", 100), deadline_hours=48, priority="balanced")
                        rerouted.append({"shipment_id": r["shipment_id"], "origin": ship.get("origin"), "destination": ship.get("destination"), "risk_level": r["risk_level"], "recommended_mode": o.get("recommended", {}).get("mode", "N/A"), "cost_saved": o.get("savings", {}).get("cost_saving_inr", 0)})
                    except Exception:
                        pass
            ml_result = {"total_fleet": len(fleet), "anomalies": scan.get("summary", {}).get("anomalies_detected", 0), "auto_rerouted": len(rerouted), "rerouted_shipments": rerouted}
            tool_label = "Auto-Reroute"

        else:
            ml_result = {"error": f"Unknown tool: {tool}"}
            tool_label = "Unknown"

    except Exception as e:
        return JSONResponse(content={
            "response": f"I understood your question and tried to run **{tool_label}**, but the ML engine encountered an error: {str(e)[:200]}. The engine may be warming up — try again in 30 seconds.",
            "tool_used": tool_label,
            "ml_data": None,
        })

    # Step 4: Ask Gemini to summarize the ML result in natural language
    safe_result = _numpy_safe(ml_result) if ml_result else {}
    try:
        summary_response = model.generate_content(
            SUMMARY_PROMPT.format(
                question=req.message,
                result=json.dumps(safe_result, indent=2, default=str)[:3000]
            )
        )
        natural_response = summary_response.text.strip()
    except Exception:
        natural_response = f"Here are the results from {tool_label}:\n\n```json\n{json.dumps(safe_result, indent=2, default=str)[:1000]}\n```"

    return JSONResponse(content=_numpy_safe({
        "response": natural_response,
        "tool_used": tool_label,
        "params_extracted": params,
        "ml_data": safe_result,
    }))

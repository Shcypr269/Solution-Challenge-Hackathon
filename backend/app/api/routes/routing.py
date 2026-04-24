from fastapi import APIRouter
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))
from ml.tomtom_traffic import compare_corridor_routes, calculate_route, INDIA_CORRIDORS

router = APIRouter()

@router.get("/alternatives/{shipment_id}")
async def get_route_alternatives(shipment_id: str, corridor_id: str = "mumbai_delhi"):
    """
    Returns real-time route alternatives using TomTom API.
    """
    if corridor_id not in INDIA_CORRIDORS:
        corridor_id = "mumbai_delhi"
        
    tomtom_result = compare_corridor_routes(corridor_id)
    
    if "error" in tomtom_result or "with_traffic" not in tomtom_result:
        # Fallback
        return {
            "shipment_id": shipment_id,
            "original_route": {
                "route_name": "NH48 via Pune (Fallback)",
                "estimated_time_mins": 240,
                "distance_km": 1400
            },
            "alternatives": []
        }
        
    traffic_route = tomtom_result["with_traffic"]
    impact = tomtom_result.get("traffic_impact", {})
    
    alternatives = [
        {
            "route_name": f"Alternative Bypass (Traffic Aware)",
            "estimated_time_mins": traffic_route['travel_time_mins'],
            "distance_km": traffic_route['distance_km'],
            "added_fuel_cost_usd": round(impact.get("extra_time_mins", 0) * 0.5, 2),
            "risk_score": 0.1,
            "incidents_on_route": traffic_route['traffic_incidents_on_route']
        }
    ]
    
    if "free_flow" in tomtom_result:
        ff_route = tomtom_result["free_flow"]
        alternatives.append({
            "route_name": "State Highway (Longer but clearer)",
            "estimated_time_mins": ff_route['travel_time_mins'] * 1.15,
            "distance_km": ff_route['distance_km'] * 1.1,
            "added_fuel_cost_usd": 15.0,
            "risk_score": 0.25,
            "incidents_on_route": 0
        })

    return {
        "shipment_id": shipment_id,
        "original_route": {
            "route_name": tomtom_result["corridor"],
            "estimated_time_mins": traffic_route['travel_time_mins'] + impact.get("extra_time_mins", 30),
            "distance_km": traffic_route['distance_km']
        },
        "alternatives": alternatives
    }


"""
Traffic API — FastAPI endpoints for TomTom traffic data.
Exposes traffic flow, route calculation, and incident queries to the Streamlit frontend.
"""
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
from dataclasses import asdict
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

router = APIRouter()


# ── Request Models ──

class TrafficFlowRequest(BaseModel):
    lat: float
    lng: float
    zoom: int = 10


class RouteRequest(BaseModel):
    origin: List[float] = Field(description="[lat, lng]")
    destination: List[float] = Field(description="[lat, lng]")
    via: Optional[List[List[float]]] = None
    traffic: bool = True
    vehicle_weight_kg: Optional[int] = None


class IncidentRequest(BaseModel):
    bbox: List[float] = Field(description="[min_lat, min_lng, max_lat, max_lng]")


class CorridorScanRequest(BaseModel):
    corridor_ids: List[str]


# ── Endpoints ──

@router.get("/corridors")
async def get_corridors():
    """Return the list of available Indian logistics corridors."""
    from ml.tomtom_traffic import INDIA_CORRIDORS, API_KEY
    corridors = {}
    for cid, data in INDIA_CORRIDORS.items():
        corridors[cid] = {
            "name": data["name"],
            "origin": list(data["origin"]),
            "destination": list(data["destination"]),
            "via": [list(v) for v in data.get("via", [])],
        }
    return {"corridors": corridors, "api_key_present": bool(API_KEY)}


@router.post("/flow")
async def get_traffic_flow_endpoint(req: TrafficFlowRequest):
    """Get real-time traffic flow at a geographic point."""
    from ml.tomtom_traffic import get_traffic_flow
    flow = get_traffic_flow(req.lat, req.lng, req.zoom)
    if flow:
        return {
            "road_name": flow.road_name,
            "current_speed_kmh": flow.current_speed_kmh,
            "free_flow_speed_kmh": flow.free_flow_speed_kmh,
            "current_travel_time_sec": flow.current_travel_time_sec,
            "free_flow_travel_time_sec": flow.free_flow_travel_time_sec,
            "confidence": flow.confidence,
            "congestion_ratio": flow.congestion_ratio,
            "road_closure": flow.road_closure,
            "congestion_level": flow.congestion_level,
            "coordinates": flow.coordinates,
        }
    return {"error": "Could not fetch traffic flow data"}


@router.post("/route")
async def calculate_route_endpoint(req: RouteRequest):
    """Calculate a traffic-aware route between two points."""
    from ml.tomtom_traffic import calculate_route
    via_tuples = [tuple(v) for v in req.via] if req.via else None
    route = calculate_route(
        origin=tuple(req.origin),
        destination=tuple(req.destination),
        via=via_tuples,
        traffic=req.traffic,
        vehicle_weight_kg=req.vehicle_weight_kg,
    )
    if route:
        return {
            "distance_km": route.distance_km,
            "travel_time_mins": route.travel_time_mins,
            "traffic_delay_mins": route.traffic_delay_mins,
            "departure_time": route.departure_time,
            "arrival_time": route.arrival_time,
            "summary": route.summary,
            "points": route.points,
            "sections": route.sections,
            "traffic_incidents_on_route": route.traffic_incidents_on_route,
        }
    return {"error": "Could not calculate route"}


@router.post("/incidents")
async def get_traffic_incidents_endpoint(req: IncidentRequest):
    """Get live traffic incidents within a bounding box."""
    from ml.tomtom_traffic import get_traffic_incidents
    incidents = get_traffic_incidents(tuple(req.bbox))
    return {
        "total": len(incidents),
        "incidents": [
            {
                "incident_id": inc.incident_id,
                "category": inc.category,
                "severity": inc.severity,
                "description": inc.description,
                "from_location": inc.from_location,
                "to_location": inc.to_location,
                "lat": inc.lat,
                "lng": inc.lng,
                "start_time": inc.start_time,
                "end_time": inc.end_time,
                "delay_seconds": inc.delay_seconds,
                "length_meters": inc.length_meters,
                "road_numbers": inc.road_numbers,
            }
            for inc in incidents
        ]
    }

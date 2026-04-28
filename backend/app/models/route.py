from pydantic import BaseModel
from typing import List, Dict

class RouteWaypoint(BaseModel):
    location: Dict[str, float]
    address: str

class RouteAlternative(BaseModel):
    route_id: str
    waypoints: List[RouteWaypoint]
    estimated_time_mins: int
    distance_km: float
    added_fuel_cost: float
    risk_score: float # 0.0 to 1.0

class RoutingStrategy(BaseModel):
    original_route_id: str
    alternatives: List[RouteAlternative]
    recommended_route_id: str

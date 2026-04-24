import os
import requests
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

API_KEY = os.environ.get("TOMTOM_API_KEY", "")
if not API_KEY:
    for env_path in [".env", "backend/.env", "../backend/.env"]:
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("TOMTOM_API_KEY="):
                        API_KEY = line.strip().split("=", 1)[1]
                        break
            if API_KEY:
                break

BASE_URL = "https://api.tomtom.com"

@dataclass
class TrafficFlowSegment:

    road_name: str
    current_speed_kmh: float
    free_flow_speed_kmh: float
    current_travel_time_sec: float
    free_flow_travel_time_sec: float
    confidence: float
    congestion_ratio: float
    road_closure: bool
    coordinates: Dict[str, float]

    @property
    def congestion_level(self) -> str:
        if self.road_closure:
            return "CLOSED"
        if self.congestion_ratio > 0.7:
            return "SEVERE"
        if self.congestion_ratio > 0.4:
            return "MODERATE"
        if self.congestion_ratio > 0.15:
            return "LIGHT"
        return "FREE_FLOW"

@dataclass
class TrafficIncident:

    incident_id: str
    category: str
    severity: int
    description: str
    from_location: str
    to_location: str
    lat: float
    lng: float
    start_time: str
    end_time: str
    delay_seconds: int
    length_meters: int
    road_numbers: List[str]

@dataclass
class RouteResult:

    distance_km: float
    travel_time_mins: float
    traffic_delay_mins: float
    departure_time: str
    arrival_time: str
    summary: str
    points: List[Dict[str, float]]
    sections: List[Dict[str, Any]]
    traffic_incidents_on_route: int

INDIA_CORRIDORS = {
    "mumbai_delhi": {
        "name": "Mumbai-Delhi (NH-48)",
        "origin": (19.0760, 72.8777),
        "destination": (28.6139, 77.2090),
        "via": [(23.0225, 72.5714),],
    },
    "delhi_kolkata": {
        "name": "Delhi-Kolkata (NH-19/NH-2)",
        "origin": (28.6139, 77.2090),
        "destination": (22.5726, 88.3639),
        "via": [(25.4358, 81.8463),],
    },
    "chennai_bangalore": {
        "name": "Chennai-Bangalore (NH-48)",
        "origin": (13.0827, 80.2707),
        "destination": (12.9716, 77.5946),
        "via": [],
    },
    "mumbai_pune": {
        "name": "Mumbai-Pune Expressway",
        "origin": (19.0760, 72.8777),
        "destination": (18.5204, 73.8567),
        "via": [],
    },
    "delhi_jaipur": {
        "name": "Delhi-Jaipur (NH-48)",
        "origin": (28.6139, 77.2090),
        "destination": (26.9124, 75.7873),
        "via": [],
    },
    "kolkata_chennai": {
        "name": "Kolkata-Chennai (NH-16)",
        "origin": (22.5726, 88.3639),
        "destination": (13.0827, 80.2707),
        "via": [(17.6868, 83.2185),],
    },
    "jnpt_nhava": {
        "name": "JNPT Port Access",
        "origin": (18.9543, 72.9486),
        "destination": (19.0760, 72.8777),
        "via": [],
    },
    "mundra_ahmedabad": {
        "name": "Mundra Port-Ahmedabad",
        "origin": (22.8386, 69.7193),
        "destination": (23.0225, 72.5714),
        "via": [],
    },
}

def get_traffic_flow(lat: float, lng: float, zoom: int = 10) -> Optional[TrafficFlowSegment]:

    url = f"{BASE_URL}/traffic/services/4/flowSegmentData/absolute/{zoom}/json"
    params = {
        "key": API_KEY,
        "point": f"{lat},{lng}",
        "unit": "KMPH",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        flow = data.get("flowSegmentData", {})

        current_speed = flow.get("currentSpeed", 0)
        free_flow = flow.get("freeFlowSpeed", 1)
        congestion = 1 - (current_speed / max(free_flow, 1))

        return TrafficFlowSegment(
            road_name=flow.get("frc", "Unknown Road"),
            current_speed_kmh=current_speed,
            free_flow_speed_kmh=free_flow,
            current_travel_time_sec=flow.get("currentTravelTime", 0),
            free_flow_travel_time_sec=flow.get("freeFlowTravelTime", 0),
            confidence=flow.get("confidence", 0),
            congestion_ratio=round(max(0, congestion), 3),
            road_closure=flow.get("roadClosure", False),
            coordinates={"lat": lat, "lng": lng},
        )
    except Exception as e:
        print(f"Traffic flow API error: {e}")
        return None

def scan_corridor_traffic(corridor_id: str) -> Dict[str, Any]:

    corridor = INDIA_CORRIDORS.get(corridor_id)
    if not corridor:
        return {"error": f"Unknown corridor: {corridor_id}"}

    sample_points = [corridor["origin"]] + corridor.get("via", []) + [corridor["destination"]]

    segments = []
    for lat, lng in sample_points:
        flow = get_traffic_flow(lat, lng)
        if flow:
            segments.append(asdict(flow))
        time.sleep(0.3)

    if segments:
        avg_speed = sum(s["current_speed_kmh"] for s in segments) / len(segments)
        avg_congestion = sum(s["congestion_ratio"] for s in segments) / len(segments)
        worst_segment = max(segments, key=lambda s: s["congestion_ratio"])
        any_closure = any(s["road_closure"] for s in segments)
    else:
        avg_speed = avg_congestion = 0
        worst_segment = None
        any_closure = False

    return {
        "corridor": corridor["name"],
        "corridor_id": corridor_id,
        "timestamp": datetime.now().isoformat(),
        "segments_scanned": len(segments),
        "avg_speed_kmh": round(avg_speed, 1),
        "avg_congestion": round(avg_congestion, 3),
        "congestion_level": (
            "CLOSED" if any_closure else
            "SEVERE" if avg_congestion > 0.7 else
            "MODERATE" if avg_congestion > 0.4 else
            "LIGHT" if avg_congestion > 0.15 else
            "FREE_FLOW"
        ),
        "any_road_closure": any_closure,
        "worst_segment": worst_segment,
        "segments": segments,
    }

def get_traffic_incidents(
    bbox: Tuple[float, float, float, float],
    categories: str = "Unknown,Accident,Fog,DangerousConditions,Rain,Ice,Jam,LaneClosed,RoadClosed,RoadWorks,Wind,Flooding"
) -> List[TrafficIncident]:

    min_lat, min_lng, max_lat, max_lng = bbox
    bbox_str = f"{min_lat},{min_lng},{max_lat},{max_lng}"

    url = f"{BASE_URL}/traffic/services/5/incidentDetails"
    params = {
        "key": API_KEY,
        "bbox": bbox_str,
        "fields": "{incidents{type,geometry{type,coordinates},properties{id,iconCategory,magnitudeOfDelay,events{description,code},startTime,endTime,from,to,length,delay,roadNumbers,timeValidity}}}",
        "language": "en-US",
        "categoryFilter": categories,
        "timeValidityFilter": "present",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        incidents = []
        for item in data.get("incidents", []):
            props = item.get("properties", {})
            geom = item.get("geometry", {})
            coords = geom.get("coordinates", [[0, 0]])[0] if geom.get("type") == "LineString" else [0, 0]

            if isinstance(coords, list) and len(coords) >= 2:
                if isinstance(coords[0], list):
                    lng, lat = coords[0][0], coords[0][1]
                else:
                    lng, lat = coords[0], coords[1]
            else:
                lat, lng = 0, 0

            events = props.get("events", [])
            description = events[0].get("description", "Unknown incident") if events else "Unknown incident"

            category_map = {
                0: "Unknown", 1: "Accident", 2: "Fog", 3: "DangerousConditions",
                4: "Rain", 5: "Ice", 6: "Jam", 7: "LaneClosed",
                8: "RoadClosed", 9: "RoadWorks", 10: "Wind", 11: "Flooding",
                14: "BrokenDownVehicle"
            }

            incidents.append(TrafficIncident(
                incident_id=str(props.get("id", "")),
                category=category_map.get(props.get("iconCategory", 0), "Unknown"),
                severity=props.get("magnitudeOfDelay", 0),
                description=description,
                from_location=props.get("from", ""),
                to_location=props.get("to", ""),
                lat=lat,
                lng=lng,
                start_time=props.get("startTime", ""),
                end_time=props.get("endTime", ""),
                delay_seconds=props.get("delay", 0),
                length_meters=props.get("length", 0),
                road_numbers=props.get("roadNumbers", []),
            ))

        return incidents

    except Exception as e:
        print(f"Incidents API error: {e}")
        return []

def get_india_wide_incidents() -> Dict[str, Any]:

    india_bbox = (8.0, 68.0, 37.0, 97.0)
    incidents = get_traffic_incidents(india_bbox)

    by_category = {}
    for inc in incidents:
        by_category.setdefault(inc.category, []).append(asdict(inc))

    severity_dist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for inc in incidents:
        severity_dist[inc.severity] = severity_dist.get(inc.severity, 0) + 1

    return {
        "timestamp": datetime.now().isoformat(),
        "total_incidents": len(incidents),
        "by_category": {k: len(v) for k, v in by_category.items()},
        "severity_distribution": severity_dist,
        "incidents": [asdict(i) for i in incidents[:50]],
    }

def calculate_route(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    via: List[Tuple[float, float]] = None,
    traffic: bool = True,
    alternatives: int = 0,
    vehicle_weight_kg: int = None,
) -> Optional[RouteResult]:

    locations = f"{origin[0]},{origin[1]}"
    if via:
        for v in via:
            locations += f":{v[0]},{v[1]}"
    locations += f":{destination[0]},{destination[1]}"

    url = f"{BASE_URL}/routing/1/calculateRoute/{locations}/json"
    params = {
        "key": API_KEY,
        "traffic": str(traffic).lower(),
        "travelMode": "truck" if vehicle_weight_kg else "car",
        "routeType": "fastest",
        "maxAlternatives": alternatives,
        "computeTravelTimeFor": "all",
        "sectionType": "traffic",
    }

    if vehicle_weight_kg:
        params["vehicleWeight"] = vehicle_weight_kg

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        routes = data.get("routes", [])
        if not routes:
            return None

        route = routes[0]
        summary = route.get("summary", {})
        legs = route.get("legs", [{}])

        points = []
        for leg in legs:
            for point in leg.get("points", []):
                points.append({"lat": point["latitude"], "lng": point["longitude"]})

        sections = []
        for section in route.get("sections", []):
            sections.append({
                "start_index": section.get("startPointIndex", 0),
                "end_index": section.get("endPointIndex", 0),
                "section_type": section.get("sectionType", ""),
                "traffic_severity": section.get("simpleCategory", ""),
            })

        travel_time_sec = summary.get("travelTimeInSeconds", 0)
        traffic_delay_sec = summary.get("trafficDelayInSeconds", 0)

        return RouteResult(
            distance_km=round(summary.get("lengthInMeters", 0) / 1000, 1),
            travel_time_mins=round(travel_time_sec / 60, 1),
            traffic_delay_mins=round(traffic_delay_sec / 60, 1),
            departure_time=summary.get("departureTime", ""),
            arrival_time=summary.get("arrivalTime", ""),
            summary=f"{summary.get('lengthInMeters', 0)/1000:.0f} km, {travel_time_sec/3600:.1f}h (+{traffic_delay_sec/60:.0f}min traffic)",
            points=points[:100],
            sections=sections,
            traffic_incidents_on_route=len(sections),
        )

    except Exception as e:
        print(f"Route API error: {e}")
        return None

def compare_corridor_routes(corridor_id: str) -> Dict[str, Any]:

    corridor = INDIA_CORRIDORS.get(corridor_id)
    if not corridor:
        return {"error": f"Unknown corridor: {corridor_id}"}

    origin = corridor["origin"]
    destination = corridor["destination"]
    via = corridor.get("via", [])

    with_traffic = calculate_route(origin, destination, via=via if via else None, traffic=True)
    no_traffic = calculate_route(origin, destination, via=via if via else None, traffic=False)

    result = {
        "corridor": corridor["name"],
        "corridor_id": corridor_id,
        "timestamp": datetime.now().isoformat(),
    }

    if with_traffic:
        result["with_traffic"] = asdict(with_traffic)
    if no_traffic:
        result["free_flow"] = asdict(no_traffic)

    if with_traffic and no_traffic:
        result["traffic_impact"] = {
            "extra_time_mins": round(with_traffic.travel_time_mins - no_traffic.travel_time_mins, 1),
            "delay_percentage": round(
                (with_traffic.travel_time_mins - no_traffic.travel_time_mins) / max(no_traffic.travel_time_mins, 1) * 100, 1
            ),
        }

    return result

if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: No TomTom API key found. Set TOMTOM_API_KEY in .env")
        exit(1)

    print("  TomTom Real-Time Traffic Integration")

    print("\sample_size--- [1] Traffic Flow: Mumbai ---")
    flow = get_traffic_flow(19.0760, 72.8777)
    if flow:
        print(f"  Road: {flow.road_name}")
        print(f"  Speed: {flow.current_speed_kmh} km/h (free-flow: {flow.free_flow_speed_kmh} km/h)")
        print(f"  Congestion: {flow.congestion_ratio:.0%} [{flow.congestion_level}]")
        print(f"  Road closure: {flow.road_closure}")

    print("\sample_size--- [2] Traffic Flow: Delhi ---")
    flow2 = get_traffic_flow(28.6139, 77.2090)
    if flow2:
        print(f"  Speed: {flow2.current_speed_kmh} km/h (free-flow: {flow2.free_flow_speed_kmh} km/h)")
        print(f"  Congestion: {flow2.congestion_ratio:.0%} [{flow2.congestion_level}]")

    print("\sample_size--- [3] Route: Mumbai -> Pune ---")
    route = calculate_route((19.0760, 72.8777), (18.5204, 73.8567), traffic=True)
    if route:
        print(f"  Distance: {route.distance_km} km")
        print(f"  Time: {route.travel_time_mins:.0f} min")
        print(f"  Traffic delay: +{route.traffic_delay_mins:.0f} min")
        print(f"  Summary: {route.summary}")

    print("\sample_size--- [4] Route: Delhi -> Jaipur ---")
    route2 = calculate_route((28.6139, 77.2090), (26.9124, 75.7873), traffic=True)
    if route2:
        print(f"  Distance: {route2.distance_km} km")
        print(f"  Time: {route2.travel_time_mins:.0f} min")
        print(f"  Traffic delay: +{route2.traffic_delay_mins:.0f} min")

    print("\sample_size--- [5] Incidents near Mumbai ---")
    mumbai_bbox = (18.5, 72.5, 19.5, 73.5)
    incidents = get_traffic_incidents(mumbai_bbox)
    print(f"  Found {len(incidents)} incidents")
    for inc in incidents[:5]:
        print(f"    [{inc.category}] {inc.description} (delay: {inc.delay_seconds//60}min)")

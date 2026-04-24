from app.agents.state import AgentState
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from ml.tomtom_traffic import compare_corridor_routes

async def routing_agent_node(state: AgentState) -> dict:
    print("Agent 3: Optimizing route via TomTom API...")
    
    pred = state.get("delay_prediction", {})
    if not pred.get("is_delayed", False) and state.get("severity_level") not in ["HIGH", "CRITICAL"]:
        print("No severe delay predicted. Route remains optimal.")
        return {
            "route_alternatives": []
        }
    
    # Pick a corridor to demo based on region or randomly
    corridor_id = "mumbai_delhi" if state.get("severity_level") == "CRITICAL" else "chennai_bangalore"
    
    # Query real TomTom data
    print(f"Querying TomTom for corridor: {corridor_id}")
    tomtom_result = compare_corridor_routes(corridor_id)
    
    alternatives = []
    
    if "error" not in tomtom_result and "with_traffic" in tomtom_result:
        traffic_route = tomtom_result["with_traffic"]
        impact = tomtom_result.get("traffic_impact", {})
        
        alternatives.append({
            "route_name": f"{tomtom_result['corridor']} (Traffic Aware)",
            "estimated_time_mins": traffic_route['travel_time_mins'],
            "distance_km": traffic_route['distance_km'],
            "added_fuel_cost_usd": round(impact.get("extra_time_mins", 0) * 0.5, 2), # 0.5 USD per extra minute
            "risk_score": 0.1,
            "incidents_on_route": traffic_route['traffic_incidents_on_route']
        })
        
        # Add a free flow / hypothetical alternative for comparison
        if "free_flow" in tomtom_result:
            ff_route = tomtom_result["free_flow"]
            alternatives.append({
                "route_name": f"Alternative State Highway",
                "estimated_time_mins": ff_route['travel_time_mins'] * 1.15, # Slightly longer baseline
                "distance_km": ff_route['distance_km'] * 1.1,
                "added_fuel_cost_usd": 15.0,
                "risk_score": 0.25,
                "incidents_on_route": 0
            })
    else:
        # Fallback if TomTom fails
        import random
        eta_savings = random.randint(15, 60)
        extra_cost = random.randint(10, 50)
        alternatives = [{
            "route_name": "Highway B bypass (Fallback)",
            "estimated_time_mins": 240 - eta_savings,
            "distance_km": 1350,
            "added_fuel_cost_usd": extra_cost,
            "risk_score": round(random.uniform(0.1, 0.4), 2),
            "incidents_on_route": 0
        }]
    
    return {
        "route_alternatives": alternatives
    }

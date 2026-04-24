
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class TransportMode:
    name: str
    cost_per_tonne_km: float
    speed_kmh: float
    co2_per_km_kg: float
    fixed_cost: float
    min_distance_km: float
    max_weight_kg: float
    reliability: float

TRANSPORT_MODES = {
    "road_truck": TransportMode(
        name="Road (Truck)", cost_per_tonne_km=2.50, speed_kmh=40,
        co2_per_km_kg=0.12, fixed_cost=500, min_distance_km=0,
        max_weight_kg=10000, reliability=0.82
    ),
    "road_ev": TransportMode(
        name="Road (EV Van)", cost_per_tonne_km=3.50, speed_kmh=35,
        co2_per_km_kg=0.02, fixed_cost=400, min_distance_km=0,
        max_weight_kg=1000, reliability=0.85
    ),
    "rail_freight": TransportMode(
        name="Rail (Freight)", cost_per_tonne_km=1.79, speed_kmh=25,
        co2_per_km_kg=0.03, fixed_cost=2000, min_distance_km=200,
        max_weight_kg=50000, reliability=0.90
    ),
    "air_cargo": TransportMode(
        name="Air Cargo", cost_per_tonne_km=18.00, speed_kmh=500,
        co2_per_km_kg=0.60, fixed_cost=5000, min_distance_km=300,
        max_weight_kg=5000, reliability=0.95
    ),
    "waterway": TransportMode(
        name="Inland Waterway", cost_per_tonne_km=1.06, speed_kmh=12,
        co2_per_km_kg=0.02, fixed_cost=3000, min_distance_km=100,
        max_weight_kg=100000, reliability=0.75
    ),
}

def optimize_transport(
    distance_km: float,
    weight_kg: float,
    deadline_hours: float,
    priority: str = "balanced",
    weather_severity: float = 0.0
) -> Dict:

    options = []

    calc_weight_tonnes = max(weight_kg / 1000.0, 0.5)

    for mode_id, mode in TRANSPORT_MODES.items():
        if distance_km < mode.min_distance_km:
            continue
        if weight_kg > mode.max_weight_kg:
            continue

        travel_time_hrs = distance_km / mode.speed_kmh
        total_cost = mode.fixed_cost + (mode.cost_per_tonne_km * distance_km * calc_weight_tonnes)
        co2_kg = mode.co2_per_km_kg * distance_km

        reliability = mode.reliability
        if "road" in mode_id and weather_severity > 0:
            reliability *= (1 - weather_severity * 0.3)
            travel_time_hrs *= (1 + weather_severity * 0.5)

        meets_deadline = travel_time_hrs <= deadline_hours

        if priority == "cost":
            score = 1.0 / (total_cost + 1)
        elif priority == "speed":
            score = 1.0 / (travel_time_hrs + 0.1)
        elif priority == "green":
            score = 1.0 / (co2_kg + 0.01)
        else:
            cost_norm = 1 - min(total_cost / 50000, 1)
            time_norm = 1 - min(travel_time_hrs / max(deadline_hours, 1), 1)
            green_norm = 1 - min(co2_kg / 100, 1)
            rel_norm = reliability
            score = 0.30 * cost_norm + 0.30 * time_norm + 0.20 * green_norm + 0.20 * rel_norm

        options.append({
            "mode": mode.name,
            "mode_id": mode_id,
            "total_cost_inr": round(total_cost),
            "travel_time_hrs": round(travel_time_hrs, 1),
            "co2_emissions_kg": round(co2_kg, 2),
            "reliability": round(reliability, 2),
            "meets_deadline": meets_deadline,
            "score": round(score, 4),
        })

    options.sort(key=lambda x: x["score"], reverse=True)

    recommended = options[0] if options else None

    if len(options) >= 2:
        worst_cost = max(o["total_cost_inr"] for o in options)
        best_cost = recommended["total_cost_inr"]
        cost_saving = worst_cost - best_cost

        worst_co2 = max(o["co2_emissions_kg"] for o in options)
        best_co2 = recommended["co2_emissions_kg"]
        co2_saving = worst_co2 - best_co2
    else:
        cost_saving = 0
        co2_saving = 0

    return {
        "input": {
            "distance_km": distance_km,
            "weight_kg": weight_kg,
            "deadline_hours": deadline_hours,
            "priority": priority,
        },
        "recommended": recommended,
        "alternatives": options,
        "savings": {
            "cost_saving_inr": cost_saving,
            "co2_saving_kg": round(co2_saving, 2),
        }
    }

if __name__ == "__main__":
    print("  Multi-Modal Transport Optimizer - Indian Logistics")

    scenarios = [
        {"desc": "Mumbai->Delhi (1400km, 500kg, 48hr deadline)",
         "distance_value": 1400, "weight": 500, "deadline": 48, "priority": "balanced"},
        {"desc": "Local delivery (15km, 5kg, 4hr, green priority)",
         "distance_value": 15, "weight": 5, "deadline": 4, "priority": "green"},
        {"desc": "Urgent shipment (800km, 20kg, 6hr)",
         "distance_value": 800, "weight": 20, "deadline": 6, "priority": "speed"},
        {"desc": "Bulk cargo (2000km, 8000kg, 96hr, cost priority)",
         "distance_value": 2000, "weight": 8000, "deadline": 96, "priority": "cost"},
        {"desc": "Monsoon scenario (500km, 100kg, 24hr, weather=0.8)",
         "distance_value": 500, "weight": 100, "deadline": 24, "priority": "balanced", "weather": 0.8},
    ]

    for sc in scenarios:
        print(f"\sample_size--- {sc['desc']} ---")
        result = optimize_transport(
            sc["distance_value"], sc["weight"], sc["deadline"],
            sc["priority"], sc.get("weather", 0)
        )

        rec = result["recommended"]
        print(f"  RECOMMENDED: {rec['mode']}")
        print(f"    Cost: INR {rec['total_cost_inr']:,} | Time: {rec['travel_time_hrs']}h | "
              f"CO2: {rec['co2_emissions_kg']}kg | Reliability: {rec['reliability']:.0%}")

        print(f"  All options:")
        for opt in result["alternatives"]:
            flag = " <<" if opt["mode_id"] == rec["mode_id"] else ""
            deadline_flag = "OK" if opt["meets_deadline"] else "LATE"
            print(f"    {opt['mode']:20s} INR {opt['total_cost_inr']:>7,} | "
                  f"{opt['travel_time_hrs']:>5.1f}h | {opt['co2_emissions_kg']:>6.1f}kg CO2 | "
                  f"{deadline_flag:4s} | score={opt['score']:.3f}{flag}")

        print(f"  Savings vs worst: INR {result['savings']['cost_saving_inr']:,}, "
              f"{result['savings']['co2_saving_kg']}kg CO2")

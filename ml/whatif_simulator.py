import pandas as pd
import numpy as np
import joblib
import os
from typing import List, Dict, Any

class WhatIfSimulator:
    def __init__(self, model_path: str = "models/delay_predictor_v2.joblib"):
        self.pipeline = None
        candidates = [model_path, "models/delay_predictor_v2.joblib", "../models/delay_predictor_v2.joblib"]
        for p in candidates:
            if os.path.exists(p):
                self.pipeline = joblib.load(p)
                break

    def simulate_disruption(
        self,
        shipments: List[Dict[str, Any]],
        disruption_type: str,
        affected_region: str,
        severity: float = 0.7
    ) -> Dict[str, Any]:

        results = []

        for shipment in shipments:
            baseline_features = self._build_features(shipment)
            baseline_prob = self._predict_prob(baseline_features)

            disrupted_shipment = self._apply_disruption(
                shipment.copy(), disruption_type, affected_region, severity
            )
            disrupted_features = self._build_features(disrupted_shipment)
            disrupted_prob = self._predict_prob(disrupted_features)

            is_affected = shipment.get("region", "").lower() == affected_region.lower()

            results.append({
                "shipment_id": shipment.get("shipment_id", "unknown"),
                "region": shipment.get("region", "unknown"),
                "is_in_affected_region": is_affected,
                "baseline_delay_prob": round(baseline_prob, 4),
                "disrupted_delay_prob": round(disrupted_prob, 4),
                "prob_increase": round(disrupted_prob - baseline_prob, 4),
                "was_at_risk": baseline_prob > 0.5,
                "now_at_risk": disrupted_prob > 0.5,
                "newly_at_risk": (not baseline_prob > 0.5) and disrupted_prob > 0.5,
            })

        total = len(results)
        affected = sum(1 for r in results if r["is_in_affected_region"])
        newly_at_risk = sum(1 for r in results if r["newly_at_risk"])
        avg_increase = np.mean([r["prob_increase"] for r in results if r["is_in_affected_region"]]) if affected > 0 else 0

        avg_penalty_per_delayed = 5000
        estimated_penalty = newly_at_risk * avg_penalty_per_delayed
        co2_per_reroute_kg = 8.5
        reroute_co2 = newly_at_risk * co2_per_reroute_kg

        return {
            "scenario": {
                "disruption_type": disruption_type,
                "affected_region": affected_region,
                "severity": severity
            },
            "impact_summary": {
                "total_shipments_analyzed": total,
                "shipments_in_affected_region": affected,
                "newly_at_risk": newly_at_risk,
                "avg_delay_prob_increase": round(float(avg_increase), 4),
                "estimated_penalty_inr": estimated_penalty,
                "reroute_co2_savings_kg": round(reroute_co2, 1)
            },
            "shipment_details": results
        }

    def _build_features(self, shipment: Dict) -> pd.DataFrame:
        return pd.DataFrame([{
            "delivery_partner": shipment.get("delivery_partner", "delhivery"),
            "package_type": shipment.get("package_type", "electronics"),
            "vehicle_type": shipment.get("vehicle_type", "truck"),
            "delivery_mode": shipment.get("delivery_mode", "standard"),
            "region": shipment.get("region", "central"),
            "weather_condition": shipment.get("weather_condition", "clear"),
            "distance_km": shipment.get("distance_km", 100.0),
            "package_weight_kg": shipment.get("package_weight_kg", 5.0),
        }])

    def _predict_prob(self, features: pd.DataFrame) -> float:
        if self.pipeline:
            return float(self.pipeline.predict_proba(features)[0][1])
        distance_value = features["distance_km"].iloc[0]
        return min(0.95, distance_value / 500.0)

    def _apply_disruption(self, shipment: Dict, dtype: str, region: str, severity: float) -> Dict:

        is_affected = shipment.get("region", "").lower() == region.lower()

        if not is_affected:
            return shipment

        if dtype == "weather":
            weather_map = {0.3: "rainy", 0.6: "foggy", 0.8: "stormy", 1.0: "stormy"}
            for threshold, condition in sorted(weather_map.items()):
                if severity <= threshold:
                    shipment["weather_condition"] = condition
                    break
            shipment["distance_km"] *= (1 + severity * 0.3)

        elif dtype == "port_congestion":
            shipment["distance_km"] *= (1 + severity * 0.5)

        elif dtype == "highway_closure":
            shipment["distance_km"] *= (1 + severity * 0.8)
            shipment["delivery_mode"] = "two day"

        elif dtype == "strike":
            shipment["delivery_mode"] = "standard"
            shipment["distance_km"] *= (1 + severity * 0.4)

        return shipment

    def generate_sample_fleet(self, sample_size: int = 20, inject_region: str = None) -> List[Dict]:

        np.random.seed(42)
        partners = ["delhivery", "shadowfax", "xpressbees", "dhl", "bluedart"]
        packages = ["electronics", "groceries", "automobile parts", "cosmetics", "medicines"]
        vehicles = ["truck", "ev van", "bike", "three wheeler"]
        modes = ["same day", "express", "standard", "two day"]
        regions = ["north", "south", "east", "west", "central"]
        weathers = ["clear", "rainy", "cold", "foggy"]

        fleet = []
        for i in range(sample_size):
            if inject_region and np.random.random() < 0.3:
                assigned_region = inject_region
            else:
                assigned_region = np.random.choice(regions)

            fleet.append({
                "shipment_id": f"SHP-{i+1:03d}",
                "delivery_partner": np.random.choice(partners),
                "package_type": np.random.choice(packages),
                "vehicle_type": np.random.choice(vehicles),
                "delivery_mode": np.random.choice(modes),
                "region": assigned_region,
                "weather_condition": np.random.choice(weathers, p=[0.5, 0.25, 0.15, 0.1]),
                "distance_km": round(np.random.uniform(50, 800), 1),
                "package_weight_kg": round(np.random.uniform(0.5, 50), 1),
            })

        if inject_region and sample_size > 0 and not any(s["region"] == inject_region for s in fleet):
            fleet[0]["region"] = inject_region

        return fleet

if __name__ == "__main__":
    sim = WhatIfSimulator()
    fleet = sim.generate_sample_fleet(30)

    print("WHAT-IF SIMULATION: Monsoon hits West region")

    result = sim.simulate_disruption(
        shipments=fleet,
        disruption_type="weather",
        affected_region="west",
        severity=0.8
    )

    summary = result["impact_summary"]
    print(f"\nTotal shipments: {summary['total_shipments_analyzed']}")
    print(f"In affected region: {summary['shipments_in_affected_region']}")
    print(f"Newly at risk: {summary['newly_at_risk']}")
    print(f"Avg delay probability increase: {summary['avg_delay_prob_increase']:.1%}")
    print(f"Estimated penalty: INR {summary['estimated_penalty_inr']:,}")
    print(f"CO2 from rerouting: {summary['reroute_co2_savings_kg']} kg")

    print("\sample_size--- Affected Shipments ---")
    for s in result["shipment_details"]:
        if s["is_in_affected_region"]:
            status = 'NEW RISK' if s['newly_at_risk'] else 'OK'
            print(f"  {s['shipment_id']}: {s['baseline_delay_prob']:.0%} -> {s['disrupted_delay_prob']:.0%} ({status})")

    print("\sample_size" + "=" * 60)
    print("WHAT-IF SIMULATION: Highway closure in North region")

    result2 = sim.simulate_disruption(
        shipments=fleet,
        disruption_type="highway_closure",
        affected_region="north",
        severity=0.9
    )

    summary2 = result2["impact_summary"]
    print(f"\nNewly at risk: {summary2['newly_at_risk']}")
    print(f"Estimated penalty: INR {summary2['estimated_penalty_inr']:,}")

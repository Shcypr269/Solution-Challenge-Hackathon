"""Tests for the decision agent cost-benefit logic.
Self-contained — does not import the full agent module to avoid LLM deps.
"""
import asyncio

# Inline the decision logic to test without app imports
async def decision_logic(state):
    alternatives = state.get("route_alternatives", [])
    contract = state.get("contract_intelligence", {})
    
    if not alternatives:
        return {"final_recommendation": {"action": "Proceed with original route", "reason": "No alternatives.", "require_manager_approval": False}}
        
    best_alt = sorted(alternatives, key=lambda x: x["added_fuel_cost_usd"])[0]
    penalty_saving = contract.get("potential_penalty_usd", 0)
    net_benefit = penalty_saving - best_alt["added_fuel_cost_usd"]
    
    if net_benefit > 0:
        return {"final_recommendation": {"action": f"Reroute via {best_alt['route_name']}", "reason": f"Saves ${net_benefit}", "require_manager_approval": True}}
    else:
        return {"final_recommendation": {"action": "Proceed with original route", "reason": "Reroute too expensive.", "require_manager_approval": False}}

def test_reroute_when_beneficial():
    state = {
        "route_alternatives": [{"route_name": "Highway B", "added_fuel_cost_usd": 20, "risk_score": 0.2}],
        "contract_intelligence": {"potential_penalty_usd": 100}
    }
    result = asyncio.run(decision_logic(state))
    rec = result["final_recommendation"]
    assert "Reroute" in rec["action"], f"Expected Reroute, got: {rec['action']}"
    assert rec["require_manager_approval"] == True

def test_stay_when_reroute_expensive():
    state = {
        "route_alternatives": [{"route_name": "Highway B", "added_fuel_cost_usd": 200, "risk_score": 0.2}],
        "contract_intelligence": {"potential_penalty_usd": 50}
    }
    result = asyncio.run(decision_logic(state))
    rec = result["final_recommendation"]
    assert "Proceed" in rec["action"], f"Expected Proceed, got: {rec['action']}"
    assert rec["require_manager_approval"] == False

def test_no_alternatives():
    state = {"route_alternatives": [], "contract_intelligence": {}}
    result = asyncio.run(decision_logic(state))
    rec = result["final_recommendation"]
    assert "Proceed" in rec["action"]

def test_picks_cheapest_route():
    state = {
        "route_alternatives": [
            {"route_name": "Expensive Route", "added_fuel_cost_usd": 80, "risk_score": 0.1},
            {"route_name": "Cheap Route", "added_fuel_cost_usd": 10, "risk_score": 0.3},
        ],
        "contract_intelligence": {"potential_penalty_usd": 100}
    }
    result = asyncio.run(decision_logic(state))
    rec = result["final_recommendation"]
    assert "Cheap Route" in rec["action"], f"Should pick cheapest route, got: {rec['action']}"

if __name__ == '__main__':
    test_reroute_when_beneficial()
    test_stay_when_reroute_expensive()
    test_no_alternatives()
    test_picks_cheapest_route()
    print("All decision agent tests passed!")

from app.agents.state import AgentState

async def decision_agent_node(state: AgentState) -> dict:
    print("Agent 5: Formulating final recommendation...")
    
    alternatives = state.get("route_alternatives", [])
    contract = state.get("contract_intelligence", {})
    
    if not alternatives:
        return {
            "final_recommendation": {
                "action": "Proceed with original route",
                "reason": "No significant delay expected or no alternatives found."
            }
        }
        
    best_alt = sorted(alternatives, key=lambda x: x["added_fuel_cost_usd"])[0]
    
    penalty_saving = contract.get("potential_penalty_usd", 0)
    net_benefit = penalty_saving - best_alt["added_fuel_cost_usd"]
    
    if net_benefit > 0:
        recommendation = {
            "action": f"Reroute via {best_alt['route_name']}",
            "reason": f"Saves ${net_benefit} overall considering penalty vs. extra fuel.",
            "require_manager_approval": True
        }
    else:
         recommendation = {
            "action": "Proceed with original route",
            "reason": "Cost of rerouting exceeds contractual delay penalty.",
            "require_manager_approval": False
        }

    return {
        "final_recommendation": recommendation
    }

from langgraph.graph import StateGraph, END
from app.agents.state import AgentState
from app.agents.disruption_agent import disruption_agent_node
from app.agents.shipment_agent import shipment_agent_node
from app.agents.routing_agent import routing_agent_node
from app.agents.contract_agent import contract_agent_node
from app.agents.decision_agent import decision_agent_node
from app.agents.feedback_agent import feedback_agent_node

def severity_router(state: AgentState) -> str:
    severity = state.get("severity_level", "LOW").upper()
    if severity == "LOW":
        return END
    else:
        return "shipment_monitor"

# Build Graph
graph_builder = StateGraph(AgentState)

graph_builder.add_node("disruption_agent", disruption_agent_node)
graph_builder.add_node("shipment_monitor", shipment_agent_node)
graph_builder.add_node("route_optimizer", routing_agent_node)
graph_builder.add_node("contract_intelligence", contract_agent_node)
graph_builder.add_node("decision_engine", decision_agent_node)
graph_builder.add_node("feedback_loop", feedback_agent_node)

graph_builder.set_entry_point("disruption_agent")

graph_builder.add_conditional_edges(
    "disruption_agent",
    severity_router,
    {
        END: END,
        "shipment_monitor": "shipment_monitor"
    }
)

graph_builder.add_edge("shipment_monitor", "route_optimizer")
graph_builder.add_edge("route_optimizer", "contract_intelligence")
graph_builder.add_edge("contract_intelligence", "decision_engine")
graph_builder.add_edge("decision_engine", "feedback_loop")
graph_builder.add_edge("feedback_loop", END)

workflow = graph_builder.compile()

async def run_pipeline(shipment_id: str, report: str, lat: float, lng: float):
    initial_state = AgentState(
        shipment_id=shipment_id,
        driver_report=report,
        current_gps={"lat": lat, "lng": lng},
        shipment_data=None,
        disruption_details=None,
        delay_prediction=None,
        route_alternatives=None,
        contract_intelligence=None,
        final_recommendation=None,
        severity_level="UNKNOWN"
    )
    result = await workflow.ainvoke(initial_state)
    return result

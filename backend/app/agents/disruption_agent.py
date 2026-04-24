from app.agents.state import AgentState
from app.ml.disruption_classifier import DisruptionClassifier

classifier = DisruptionClassifier()

async def disruption_agent_node(state: AgentState) -> dict:
    print(f"Agent 1: Analyzing disruption for shipment {state['shipment_id']}")
    report = state["driver_report"]
    details = await classifier.classify_text(report)
    return {
        "disruption_details": details,
        "severity_level": details.get("severity", "MEDIUM")
    }

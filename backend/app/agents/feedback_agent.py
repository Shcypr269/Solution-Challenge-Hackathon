import json
import logging
from datetime import datetime
from app.agents.state import AgentState

logger = logging.getLogger(__name__)

# In-memory feedback store (Firestore/BigQuery in production)
_feedback_log = []

async def feedback_agent_node(state: AgentState) -> dict:
    """
    Agent 6: Feedback & Learning Agent
    Logs the full pipeline outcome for future retraining and auditing.
    """
    print("Agent 6: Logging feedback loop...")
    
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "shipment_id": state.get("shipment_id"),
        "severity_level": state.get("severity_level"),
        "disruption_details": state.get("disruption_details"),
        "delay_prediction": state.get("delay_prediction"),
        "route_alternatives_count": len(state.get("route_alternatives") or []),
        "contract_intelligence": state.get("contract_intelligence"),
        "final_recommendation": state.get("final_recommendation"),
        "outcome": "pending_manager_review"
    }
    
    _feedback_log.append(entry)
    logger.info(f"Feedback logged: {json.dumps(entry, default=str)}")
    
    return {}

def get_feedback_log():
    """Returns all logged pipeline outcomes for analysis."""
    return _feedback_log

from typing import TypedDict, Annotated, List, Dict, Any, Optional
import operator

class AgentState(TypedDict):
    # Initial input
    shipment_id: str
    driver_report: str
    current_gps: Dict[str, float]
    
    # Context injected by agents
    shipment_data: Optional[Dict[str, Any]]
    disruption_details: Optional[Dict[str, Any]]
    
    # Predictions
    delay_prediction: Optional[Dict[str, Any]]
    
    # Alternatives
    route_alternatives: Optional[List[Dict[str, Any]]]
    contract_intelligence: Optional[Dict[str, Any]]
    
    # Final Decision
    final_recommendation: Optional[Dict[str, Any]]
    
    # Routing control
    severity_level: str # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'

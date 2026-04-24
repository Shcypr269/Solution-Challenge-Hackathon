from pydantic import BaseModel
from typing import List

class ContractPenalty(BaseModel):
    condition_description: str
    penalty_amount_inr: float
    is_per_hour: bool

class ContractIntelligence(BaseModel):
    contract_id: str
    shipment_id: str
    relevant_clauses: List[str]
    potential_penalty_inr: float
    summary_of_risk: str

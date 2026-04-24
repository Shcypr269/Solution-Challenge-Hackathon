from pydantic import BaseModel, Field
from typing import Optional, Dict
from enum import Enum

class DisruptionSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class Disruption(BaseModel):
    id: str
    shipment_id: str
    event_type: str
    severity: DisruptionSeverity
    location: Dict[str, float]
    description: str
    estimated_clearance_time_mins: int = 0
    confidence: float = 1.0
    timestamp: str

class DisruptionReport(BaseModel):
    """Payload received from the driver app"""
    shipment_id: str
    driver_id: str
    voice_transcript: str
    current_gps: Dict[str, float]

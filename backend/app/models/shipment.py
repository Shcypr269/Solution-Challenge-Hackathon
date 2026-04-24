from enum import Enum
from pydantic import BaseModel, ConfigDict
from typing import Optional, Dict, Any

class ShipmentStatus(str, Enum):
    PENDING = "PENDING"
    IN_TRANSIT = "IN_TRANSIT"
    DELAYED = "DELAYED"
    DELIVERED = "DELIVERED"
    CANCELLED = "CANCELLED"

class Shipment(BaseModel):
    id: str
    tracking_number: str
    origin_city: str
    destination_city: str
    status: ShipmentStatus
    scheduled_delivery_date: str 
    current_location: Optional[Dict[str, float]] = None 
    contract_id: Optional[str] = None
    
    # Store features for delay prediction model
    ml_features: Optional[Dict[str, Any]] = None 
    model_config = ConfigDict(from_attributes=True)

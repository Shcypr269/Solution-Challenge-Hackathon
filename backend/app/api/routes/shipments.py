from fastapi import APIRouter, HTTPException
from typing import List, Optional
from app.models.shipment import Shipment, ShipmentStatus

router = APIRouter()

# In-memory store for demo (Firestore in production)
_shipments_db = {
    "SHP-001": Shipment(
        id="SHP-001",
        tracking_number="TRK-2026-001",
        origin_city="Mumbai",
        destination_city="Delhi",
        status=ShipmentStatus.IN_TRANSIT,
        scheduled_delivery_date="2026-04-22T10:00:00",
        current_location={"lat": 23.2599, "lng": 77.4126},
        ml_features={
            "delivery_partner": "delhivery",
            "package_type": "electronics",
            "vehicle_type": "truck",
            "delivery_mode": "express",
            "region": "central",
            "weather_condition": "clear",
            "distance_km": 1400,
            "package_weight_kg": 25.0
        }
    ),
    "SHP-002": Shipment(
        id="SHP-002",
        tracking_number="TRK-2026-002",
        origin_city="Bangalore",
        destination_city="Chennai",
        status=ShipmentStatus.IN_TRANSIT,
        scheduled_delivery_date="2026-04-21T18:00:00",
        current_location={"lat": 12.9716, "lng": 77.5946},
        ml_features={
            "delivery_partner": "shadowfax",
            "package_type": "groceries",
            "vehicle_type": "ev van",
            "delivery_mode": "same day",
            "region": "south",
            "weather_condition": "rainy",
            "distance_km": 350,
            "package_weight_kg": 8.0
        }
    )
}

@router.get("/", response_model=List[Shipment])
async def list_shipments():
    return list(_shipments_db.values())

@router.get("/{shipment_id}", response_model=Shipment)
async def get_shipment(shipment_id: str):
    shipment = _shipments_db.get(shipment_id)
    if not shipment:
        raise HTTPException(status_code=404, detail=f"Shipment {shipment_id} not found")
    return shipment

@router.patch("/{shipment_id}/status")
async def update_shipment_status(shipment_id: str, status: ShipmentStatus):
    shipment = _shipments_db.get(shipment_id)
    if not shipment:
        raise HTTPException(status_code=404, detail=f"Shipment {shipment_id} not found")
    shipment.status = status
    return {"message": f"Shipment {shipment_id} status updated to {status}"}

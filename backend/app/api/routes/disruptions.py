from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict
from app.agents.graph import run_pipeline

router = APIRouter()

class DisruptionPayload(BaseModel):
    shipment_id: str
    driver_report: str
    current_gps: Dict[str, float]

@router.post("/")
async def report_disruption(payload: DisruptionPayload):
    """
    Endpoint triggered by the driver app when a disruption is reported.
    It kicks off the LangGraph multi-agent pipeline.
    """
    try:
        # Run pipeline 
        result = await run_pipeline(
            shipment_id=payload.shipment_id,
            report=payload.driver_report,
            lat=payload.current_gps.get("lat", 0.0),
            lng=payload.current_gps.get("lng", 0.0)
        )
        
        return {
            "status": "Pipeline completed",
            "shipment_id": result.get("shipment_id"),
            "severity_level": result.get("severity_level"),
            "final_recommendation": result.get("final_recommendation")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

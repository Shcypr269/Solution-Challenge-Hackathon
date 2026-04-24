from app.api.routes.websocket import notify_managers

async def send_recommendation_alert(shipment_id: str, recommendation: dict):
    """
    Pushes a recommendation to all connected manager dashboards via WebSocket.
    """
    alert = {
        "type": "recommendation",
        "shipment_id": shipment_id,
        "recommendation": recommendation
    }
    await notify_managers(alert)

async def send_disruption_alert(shipment_id: str, disruption: dict):
    """
    Pushes a disruption detection alert to manager dashboards.
    """
    alert = {
        "type": "disruption_detected",
        "shipment_id": shipment_id,
        "disruption": disruption
    }
    await notify_managers(alert)

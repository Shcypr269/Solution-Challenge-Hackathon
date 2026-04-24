from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json

router = APIRouter()

# Very basic connection manager for demo 
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@router.websocket("/ws/manager")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket for pushing live alerts to the Manager Dashboard App.
    """
    await manager.connect(websocket)
    try:
        while True:
            # We just keep connection open, but Manager Dashboard listens to broadcasts
            data = await websocket.receive_text()
            # If the manager approves from dashboard, it could come back here
            message = f"Manager says: {data}"
            print(message)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def notify_managers(update_dict: dict):
    # Called by Agent 5 (Decision Engine) 
    await manager.broadcast(json.dumps(update_dict))

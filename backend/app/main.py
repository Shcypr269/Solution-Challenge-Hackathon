import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import disruptions, websocket, shipments, routing, ml_api

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Supply Chain Backend...")
    yield
    logger.info("Shutting down...")

app = FastAPI(
    title="Self-Healing Supply Chain API",
    description="Backend API for logistics disruption handling and AI agent orchestration.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "0.1.0"}

app.include_router(shipments.router, prefix="/api/v1/shipments", tags=["Shipments"])
app.include_router(disruptions.router, prefix="/api/v1/disruptions", tags=["Disruptions"])
app.include_router(routing.router, prefix="/api/v1/routes", tags=["Routing"])
app.include_router(ml_api.router, prefix="/api/v1/ml", tags=["ML Predictions"])
app.include_router(websocket.router, tags=["Realtime"])

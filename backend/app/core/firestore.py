import os
from google.cloud import firestore
from app.config import settings
from typing import Optional

db: Optional[firestore.AsyncClient] = None

def get_firestore_client() -> firestore.AsyncClient:
    global db
    if db is None:
        if os.getenv("FIRESTORE_EMULATOR_HOST"):
            db = firestore.AsyncClient(project=settings.gcp_project_id)
        else:
            db = firestore.AsyncClient(project=settings.gcp_project_id)
    return db

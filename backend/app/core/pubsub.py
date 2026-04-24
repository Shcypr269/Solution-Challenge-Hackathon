import asyncio
import os
from google.cloud import pubsub_v1
from app.config import settings

publisher = None
if os.getenv("PUBSUB_EMULATOR_HOST"):
    publisher = pubsub_v1.PublisherClient()
else:
    publisher = pubsub_v1.PublisherClient()

topic_path = publisher.topic_path(settings.gcp_project_id, "disruption_events")

async def publish_disruption_event(payload: dict):

    import json
    data = json.dumps(payload).encode("utf-8")
    
    # Run the blocking publish in an executor
    loop = asyncio.get_running_loop()
    future = await loop.run_in_executor(None, publisher.publish, topic_path, data)
    return future.result()

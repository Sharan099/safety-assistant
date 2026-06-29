import os
from celery import Celery

# Check environment for Redis connection URL
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = os.getenv("REDIS_PORT", "6379")
redis_url = os.getenv("REDIS_URL", f"redis://{redis_host}:{redis_port}/0")

celery_app = Celery(
    "safety_registry_scheduler",
    broker=redis_url,
    backend=redis_url,
    include=["scheduler.tasks"]
)

# Optional configurations
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600, # 1 hour timeout max
)

if __name__ == "__main__":
    celery_app.start()

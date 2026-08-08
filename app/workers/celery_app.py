"""
Celery application — background task queue for long-running evaluations.

CONCEPT: Why a Task Queue?
───────────────────────────
Evaluating 1000 samples × 5 models = 5000 API calls.
At 500ms/call, that's 41 minutes — way too long for an HTTP request.

The API accepts the request, enqueues a Celery task, and returns immediately
with a run_id. The client polls GET /evaluations/{run_id} for status.

Worker processes pick up tasks from Redis, run the evaluation, and write
results to PostgreSQL + MLflow. This is the standard async job pattern.

Sprint 4 will implement the actual eval tasks.
"""
from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "evalforge",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.workers.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True, # Task not acked until complete (safer)
    worker_prefetch_multiplier=1  # One task at a time per worker slot
)
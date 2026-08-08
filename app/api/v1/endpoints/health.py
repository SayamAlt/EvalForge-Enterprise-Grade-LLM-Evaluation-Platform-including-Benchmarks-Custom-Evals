"""Health check endpoint — confirms the API is alive and shows uptime."""
import time
from fastapi import APIRouter
from pydantic import BaseModel
from app.core.config import settings

router = APIRouter()
_start_time = time.time()

class HealthResponse(BaseModel):
    status: str
    app: str
    env: str
    uptime_seconds: float
    version: str

@router.get("", response_model=HealthResponse, summary="Liveness check")
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        env=settings.app_env,
        uptime_seconds=round(time.time() - _start_time, 2),
        version="0.1.0"
    )
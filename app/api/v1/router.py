"""
API v1 router — wires all endpoint modules together.

Route structure:
    GET  /api/v1/health          → liveness check
    *    /api/v1/datasets/*      → Sprint 2
    *    /api/v1/models/*        → Sprint 3
    *    /api/v1/evaluations/*   → Sprint 4
    *    /api/v1/experiments/*   → Sprint 5
"""
from fastapi import APIRouter
from app.api.v1.endpoints import datasets, evaluations, experiments, health, models, benchmarks, exports

v1_router = APIRouter()

v1_router.include_router(health.router, prefix="/health", tags=["Health"])
v1_router.include_router(datasets.router, prefix="/datasets", tags=["Datasets"])
v1_router.include_router(models.router, prefix="/models", tags=["Model Registry"])
v1_router.include_router(evaluations.router, prefix="/evaluations", tags=["Evaluations"])
v1_router.include_router(experiments.router, prefix="/experiments", tags=["Experiments"])
v1_router.include_router(benchmarks.router, prefix="/benchmarks", tags=["Benchmarks"])
v1_router.include_router(exports.router, prefix="/evaluations", tags=["Exports"])
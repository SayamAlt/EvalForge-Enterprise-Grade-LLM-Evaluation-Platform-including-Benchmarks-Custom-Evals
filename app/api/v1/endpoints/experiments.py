"""
Experiment tracking endpoints — implemented in Sprint 5.

Planned routes:
    POST   /experiments                  Create a new experiment
    GET    /experiments                  List experiments
    GET    /experiments/{id}             Experiment details + all runs
    GET    /experiments/{id}/leaderboard Model leaderboard for this experiment
    POST   /experiments/{id}/runs        Add a run to an experiment
    POST   /experiments/compare          Compare multiple runs side-by-side
"""
import uuid
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.deps import get_db
from app.schemas.experiment import (ExperimentCreate, ExperimentResponse,
    AddRunToExperiment, LeaderboardResponse, ComparisonResponse)
from app.schemas.evaluation import EvalRunSummaryResponse
from app.services import experiment_service
from app.models.experiment import ExperimentORM
from app.core.exceptions import NotFoundError

router = APIRouter()

@router.post("", response_model=ExperimentResponse, status_code=201)
async def create_experiment(body: ExperimentCreate, db: AsyncSession = Depends(get_db)):
    exp = await experiment_service.create_experiment(db, body)
    await db.commit()
    return exp

@router.get("", response_model=list[ExperimentResponse])
async def list_experiments(db: AsyncSession = Depends(get_db)):
    return await experiment_service.list_experiments(db)

@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    exp = await db.get(ExperimentORM, experiment_id)

    if not exp:
        raise HTTPException(404, detail="Experiment not found")

    return exp

@router.get("/{experiment_id}/runs", response_model=list[EvalRunSummaryResponse])
async def list_experiment_runs(experiment_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    try:
        return await experiment_service.get_experiment_runs(db, experiment_id)
    except NotFoundError as e:
        raise HTTPException(404, detail=str(e))

@router.post("/{experiment_id}/runs", status_code=204)
async def add_run(experiment_id: uuid.UUID, body: AddRunToExperiment, db: AsyncSession = Depends(get_db)):
    try:
        await experiment_service.add_run(db, experiment_id, body.run_id)
        await db.commit()
    except NotFoundError as e:
        raise HTTPException(404, detail=str(e))

@router.get("/{experiment_id}/leaderboard", response_model=LeaderboardResponse)
async def get_leaderboard(experiment_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    try:
        return await experiment_service.get_leaderboard(db, experiment_id)
    except NotFoundError as e:
        raise HTTPException(404, detail=str(e))

@router.post("/compare", response_model=ComparisonResponse)
async def compare_runs(run_ids: list[uuid.UUID] = Body(...), db: AsyncSession = Depends(get_db)):
    try:
        return await experiment_service.compare_runs(db, run_ids)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
"""
Evaluation run endpoints.

Routes:
    POST   /evaluations              Launch eval (returns immediately, runs in background)
    POST   /evaluations/batch        Launch eval across multiple models in parallel
    GET    /evaluations              List all runs (filterable by model, dataset)
    GET    /evaluations/{run_id}     Get run status + results (poll this for progress)
    GET    /evaluations/{run_id}/samples  Per-sample results
    DELETE /evaluations/{run_id}     Delete a run
"""
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.api.deps import get_db
from app.schemas.evaluation import (
    BatchEvalCreate,
    EvalRunCreate,
    EvalRunListResponse,
    EvalRunResponse,
    EvalRunSummaryResponse,
    SampleResultResponse,
)
from app.services import evaluation_service
from app.models.evaluation import EvaluationRunORM
from app.core.exceptions import NotFoundError, DatasetError, ProviderError

router = APIRouter()


@router.post("", response_model=EvalRunResponse, status_code=202)
async def create_evaluation(body: EvalRunCreate, db: AsyncSession = Depends(get_db)):
    """
    Launch an evaluation run.

    Returns immediately (status='running') — the evaluation continues in the background.
    Poll GET /evaluations/{id} to track progress via config.completed_samples / config.total_samples.
    """
    try:
        run = await evaluation_service.launch_run(db, body)
        return run
    except NotFoundError as e:
        raise HTTPException(404, detail=str(e))
    except (DatasetError, ProviderError) as e:
        raise HTTPException(400, detail=str(e))


@router.post("/batch", response_model=list[EvalRunResponse], status_code=202)
async def create_batch_evaluation(body: BatchEvalCreate, db: AsyncSession = Depends(get_db)):
    """
    Launch evaluations for multiple models against the same dataset in parallel.

    Each model gets its own run record. All run in background simultaneously.
    Returns a list of run objects (all with status='running').

    Example — run all available models on HLE:
    ```json
    {
      "dataset_id": "47cfec6f-...",
      "model_ids": ["gpt-4o", "claude-haiku-4-5", "llama-3.1-8b-groq", "gemini-2.5-flash"],
      "num_samples": 100,
      "concurrency": 5,
      "metrics": ["answer_match"]
    }
    ```
    """
    try:
        body_list = [
            EvalRunCreate(
                dataset_id=body.dataset_id,
                model_id=mid,
                num_samples=body.num_samples,
                concurrency=body.concurrency,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
                metrics=body.metrics,
            )
            for mid in body.model_ids
        ]
        runs = await evaluation_service.launch_batch(db, body_list)
        return runs
    except NotFoundError as e:
        raise HTTPException(404, detail=str(e))
    except (DatasetError, ProviderError) as e:
        raise HTTPException(400, detail=str(e))


@router.get("", response_model=EvalRunListResponse, response_model_exclude_none=False)
async def list_evaluations(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    model_id: str | None = Query(default=None),
    dataset_id: uuid.UUID | None = Query(default=None),
    status: str | None = Query(default=None, description="Filter by status: running|done|error"),
    db: AsyncSession = Depends(get_db),
):
    q = select(EvaluationRunORM).order_by(EvaluationRunORM.created_at.desc())
    if model_id:
        q = q.where(EvaluationRunORM.model_id == model_id)
    if dataset_id:
        q = q.where(EvaluationRunORM.dataset_id == dataset_id)
    if status:
        q = q.where(EvaluationRunORM.status == status)

    total_q = select(func.count()).select_from(q.subquery())
    total = (await db.execute(total_q)).scalar_one()
    rows = (await db.execute(q.offset((page - 1) * page_size).limit(page_size))).scalars().all()
    return EvalRunListResponse(items=list(rows), total=total, page=page, page_size=page_size)


@router.get("/{run_id}", response_model=EvalRunResponse)
async def get_evaluation(run_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """
    Get evaluation run status and results.

    While status='running', check config.completed_samples / config.total_samples for progress.
    """
    run = await db.get(EvaluationRunORM, run_id)
    if not run:
        raise HTTPException(404, detail=f"Run {run_id} not found")
    return run


@router.get("/{run_id}/samples", response_model=list[SampleResultResponse])
async def get_evaluation_samples(run_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    run = await db.get(EvaluationRunORM, run_id)
    if not run:
        raise HTTPException(404, detail=f"Run {run_id} not found")
    if not run.sample_results:
        return []
    return [SampleResultResponse(**s) for s in run.sample_results]


@router.delete("/{run_id}", status_code=204)
async def delete_evaluation(run_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    run = await db.get(EvaluationRunORM, run_id)
    if not run:
        raise HTTPException(404, detail=f"Run {run_id} not found")
    await db.delete(run)
    await db.commit()

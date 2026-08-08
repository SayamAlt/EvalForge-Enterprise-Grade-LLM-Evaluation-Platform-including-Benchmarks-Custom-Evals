import csv, io, json, uuid
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.deps import get_db
from app.models.evaluation import EvaluationRunORM

router = APIRouter()

@router.get("/{run_id}/export/csv")
async def export_csv(run_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    run = await db.get(EvaluationRunORM, run_id)

    if not run:
        raise HTTPException(404, detail="Run not found")

    samples = run.sample_results or []

    if not samples:
        raise HTTPException(400, detail="Run has no sample results")

    metric_keys = list(samples[0].get("metrics", {}).keys())
    fieldnames = [
        "sample_id", "input", "reference", "prediction",
        "latency_ms", "input_tokens", "output_tokens", "cost_usd", "error"
    ] + metric_keys

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    for s in samples:
        row = {**s, **s.get("metrics", {})}
        writer.writerow(row)

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=run_{run_id}.csv"}
    )

@router.get("/{run_id}/export/json")
async def export_json(run_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    run = await db.get(EvaluationRunORM, run_id)

    if not run:
        raise HTTPException(404, detail="Run not found")

    data = {
        "run_id": str(run.id),
        "model_id": run.model_id,
        "provider": run.provider,
        "benchmark_name": run.benchmark_name,
        "status": run.status,
        "aggregate_metrics": run.aggregate_metrics,
        "sample_results": run.sample_results,
        "created_at": run.created_at.isoformat(),
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
    }

    return StreamingResponse(
        iter([json.dumps(data, indent=2, default=str)]),
        media_type="application/json",
        headers={"Content-Disposition": f"attachment; filename=run_{run_id}.json"}
    )
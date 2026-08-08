import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.deps import get_db
from app.schemas.benchmark import BenchmarkInfo, BenchmarkRunCreate
from app.schemas.evaluation import EvalRunResponse
from evalforge.datasets.benchmarks import BENCHMARK_CATALOG, load_benchmark
from evalforge.datasets.base import SimpleDataset
from evalforge.providers.registry import get_provider
from evalforge.metrics.text import get_metrics
from evalforge.evaluators.custom_evaluator import CustomEvaluator
from app.models.evaluation import EvaluationRunORM
from app.api.v1.endpoints.models import MODEL_CATALOG
from app.services import experiment_service
from app.services.evaluation_service import save_leaderboard_csv

router = APIRouter()

TASK_METRICS = {
    "multiple_choice": "choice_match",
    "generation": "answer_match",
    "code": "code_match",
    "classification": "label_match",
}

@router.get("", response_model=list[BenchmarkInfo])
async def list_benchmarks():
    return [
        BenchmarkInfo(
            name=k,
            hf_id=v["hf_id"],
            task_type=v["task_type"],
            default_metric=v["default_metric"],
            needs_subject=v.get("needs_config", False)
        )
        for k, v in BENCHMARK_CATALOG.items()
    ]

@router.post("/run", response_model=EvalRunResponse, status_code=201)
async def run_benchmark(body: BenchmarkRunCreate, db: AsyncSession = Depends(get_db)):
    entry = MODEL_CATALOG.get(body.model_id)

    if not entry:
        raise HTTPException(404, detail=f"Model '{body.model_id}' not found")

    if body.benchmark not in BENCHMARK_CATALOG:
        raise HTTPException(400, detail=f"Unknown benchmark '{body.benchmark}'")

    cfg = BENCHMARK_CATALOG[body.benchmark]

    if cfg.get("needs_config") and not body.subject:
        raise HTTPException(400, detail=f"Benchmark '{body.benchmark}' requires a subject")

    samples = await load_benchmark(body.benchmark, body.subject, body.split, body.num_samples)
    metric_name = cfg["default_metric"]
    dataset_name = f"{body.benchmark}_{body.subject}" if body.subject else body.benchmark

    run = EvaluationRunORM(
        model_id=body.model_id,
        provider=entry.provider,
        status="running",
        benchmark_name=dataset_name,
        config={
            "benchmark": body.benchmark,
            "subject": body.subject,
            "split": body.split,
            "metrics": [metric_name],
            "num_samples": body.num_samples
        },
    )
    db.add(run)
    await db.flush()

    dataset = SimpleDataset(name=dataset_name, samples=samples)
    provider = get_provider(entry.provider, entry.provider_model_id)
    evaluator = CustomEvaluator(
        provider=provider,
        metrics=get_metrics([metric_name]),
        concurrency=body.concurrency,
    )
    report = await evaluator.run(dataset, run_id=str(run.id), num_samples=body.num_samples)

    run.status = "done"
    run.num_samples = report.num_samples
    run.num_errors = report.n_errors
    run.error_rate = report.error_rate
    run.avg_latency_ms = report.avg_latency_ms
    run.total_cost_usd = report.total_cost_usd
    run.duration_seconds = report.duration_seconds
    run.aggregate_metrics = {
        k: {"mean": v.mean, "std": v.std, "min": v.min, "max": v.max, "n": v.n}
        for k, v in report.aggregate_metrics.items()
    }
    run.sample_results = [vars(r) for r in report.sample_results]
    run.completed_at = datetime.now(timezone.utc)
    await db.flush()

    if body.experiment_id:
        await experiment_service.add_run(db, uuid.UUID(body.experiment_id), run.id)

    await save_leaderboard_csv(run, cfg.get("task_type", "benchmark"), dataset_name, db)
    await db.commit()
    return run
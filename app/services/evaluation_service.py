import asyncio
import csv
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from sqlalchemy.orm.attributes import flag_modified
from app.core.database import get_db_session
from app.core.exceptions import DatasetError, NotFoundError, ProviderError
from app.core.logging import get_logger
from app.models.dataset import DatasetORM
from app.models.evaluation import EvaluationRunORM
from app.schemas.evaluation import EvalRunCreate
from evalforge.datasets.base import SimpleDataset
from evalforge.datasets.loaders.csv_loader import load_csv, load_jsonl
from evalforge.evaluators.custom_evaluator import CustomEvaluator
from evalforge.metrics.text import get_metrics
from evalforge.providers.registry import get_provider

RESULTS_DIR = Path(__file__).parents[2] / "results"
BATCH_SIZE = 100  # samples committed to DB at a time

_TASK_DEFAULT_METRIC = {
    "qa": ["answer_match"],
    "multiple_choice": ["choice_match"],
    "generation": ["answer_match"],
    "code": ["code_match"],
    "classification": ["label_match"],
}

# Dataset-specific overrides — take precedence over task-level defaults
_DATASET_DEFAULT_METRIC: dict[str, list[str]] = {
    "gsm8k": ["numeric_match"],
    "openai/gsm8k": ["numeric_match"],
}

# Optimal concurrency per provider — tuned to their rate limits
_PROVIDER_CONCURRENCY: dict[str, int] = {
    "openai": 15,
    "anthropic": 5,
    "google": 10,
    "groq": 10,
    "mistral": 8,
    "deepseek": 8,
    "ollama": 4,
}

logger = get_logger(__name__)


def _provider_concurrency(provider: str, requested: int) -> int:
    """Use the lower of requested and provider's safe ceiling."""
    ceiling = _PROVIDER_CONCURRENCY.get(provider, 10)
    return min(requested, ceiling)


async def save_leaderboard_csv(
    run: EvaluationRunORM,
    task_type: str,
    display_name: str,
    db: AsyncSession,
) -> None:
    """Write/update leaderboard CSV for all done runs of the same dataset or benchmark."""
    if run.dataset_id:
        stmt = select(EvaluationRunORM).where(
            EvaluationRunORM.dataset_id == run.dataset_id,
            EvaluationRunORM.status == "done",
        )
    else:
        stmt = select(EvaluationRunORM).where(
            EvaluationRunORM.benchmark_name == run.benchmark_name,
            EvaluationRunORM.status == "done",
        )

    result = await db.execute(stmt)
    all_runs = result.scalars().all()

    primary_metric = ((run.config or {}).get("metrics") or ["score"])[0]

    # Keep only the best run per model (highest score, then most recent)
    best: dict[str, EvaluationRunORM] = {}
    for r in all_runs:
        score = (r.aggregate_metrics or {}).get(primary_metric, {}).get("mean", 0.0)
        prev = best.get(r.model_id)
        if prev is None:
            best[r.model_id] = r
        else:
            prev_score = (prev.aggregate_metrics or {}).get(primary_metric, {}).get("mean", 0.0)
            if score > prev_score:
                best[r.model_id] = r

    rows = []
    for r in best.values():
        score = (r.aggregate_metrics or {}).get(primary_metric, {}).get("mean", 0.0)
        cost_per_sample = (r.total_cost_usd / r.num_samples) if r.num_samples else 0.0
        rows.append({
            "model_id": r.model_id,
            "provider": r.provider,
            primary_metric: round(score, 4),
            "avg_latency_ms": round(r.avg_latency_ms or 0, 2),
            "cost_per_sample_usd": round(cost_per_sample, 7),
            "num_samples": r.num_samples,
            "error_rate": round(r.error_rate or 0, 4),
        })

    rows.sort(key=lambda x: x[primary_metric], reverse=True)
    for i, row in enumerate(rows, 1):
        row["rank"] = i

    slug = display_name.lower().replace(" ", "_").replace("-", "_")
    out_dir = RESULTS_DIR / task_type
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{slug}_leaderboard.csv"

    fieldnames = ["rank", "model_id", "provider", primary_metric,
                  "avg_latency_ms", "cost_per_sample_usd", "num_samples", "error_rate"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("leaderboard.written", path=str(csv_path), models=len(rows))


async def _load_samples(dataset_row: DatasetORM, n_samples: int | None) -> list:
    """Load samples from HuggingFace or file-based dataset."""
    if dataset_row.source_type == "huggingface":
        from evalforge.datasets.loaders.huggingface_loader import load_from_huggingface
        return await asyncio.to_thread(
            load_from_huggingface,
            hf_dataset_id=dataset_row.hf_dataset_id,
            dataset_name=dataset_row.name,
            hf_config=dataset_row.hf_config,
            hf_split=dataset_row.hf_split or "test",
            n_samples=n_samples,
            format_config=dataset_row.format_config,
        )
    elif dataset_row.source_type == "csv":
        samples = load_csv(Path(dataset_row.file_path).read_bytes(), dataset_row.name)
    else:
        samples = load_jsonl(Path(dataset_row.file_path).read_bytes(), dataset_row.name)
    return samples[:n_samples] if n_samples else samples


async def _run_eval_task(
    run_id: uuid.UUID,
    dataset_id: uuid.UUID,
    chosen_metrics: list[str],
    provider_name: str,
    provider_model_id: str,
    concurrency: int,
    temperature: float,
    max_tokens: int,
    n_samples: int | None,
) -> None:
    """
    Background coroutine: runs a full evaluation and writes results in batches.
    Uses its own DB sessions — safe to run after the HTTP request completes.
    """
    import time
    from evalforge.evaluators.base_evaluator import SampleResult

    try:
        # Load dataset info + samples
        async with get_db_session() as db:
            dataset_row = await db.get(DatasetORM, dataset_id)
            if not dataset_row:
                raise NotFoundError(f"Dataset {dataset_id} not found")
            task_type = dataset_row.task_type
            dataset_name = dataset_row.name

        samples = await _load_samples(dataset_row, n_samples)

        # Update total sample count
        async with get_db_session() as db:
            run = await db.get(EvaluationRunORM, run_id)
            run.num_samples = len(samples)
            run.config = {**(run.config or {}), "total_samples": len(samples), "completed_samples": 0}
            flag_modified(run, "config")

        provider = get_provider(provider_name, provider_model_id)
        metrics = get_metrics(chosen_metrics)
        effective_concurrency = _provider_concurrency(provider_name, concurrency)
        evaluator = CustomEvaluator(
            provider=provider,
            metrics=metrics,
            concurrency=effective_concurrency,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        all_results: list[SampleResult] = []
        start = time.monotonic()

        # Process in batches — intermediate DB commits for progress + fault tolerance
        for batch_start in range(0, len(samples), BATCH_SIZE):
            batch = samples[batch_start: batch_start + BATCH_SIZE]
            batch_results: list[SampleResult] = list(await asyncio.gather(
                *[evaluator._evaluate_sample_with_retry(s) for s in batch]
            ))
            all_results.extend(batch_results)

            # Fast-abort: if first batch has >80% fatal errors, stop immediately
            if batch_start == 0 and batch_results:
                fatal_rate = sum(
                    1 for r in batch_results
                    if r.error and any(p in (r.error or "").lower() for p in (
                        "credit balance", "billing", "insufficient", "authentication",
                        "invalid api key", "incorrect api key",
                    ))
                ) / len(batch_results)
                if fatal_rate > 0.8:
                    raise RuntimeError(
                        f"Aborting: {fatal_rate:.0%} of first batch returned fatal errors — "
                        f"{batch_results[0].error}"
                    )

            # Flush progress after each batch
            completed = len(all_results)
            async with get_db_session() as db:
                run = await db.get(EvaluationRunORM, run_id)
                run.config = {**(run.config or {}), "completed_samples": completed}
                flag_modified(run, "config")

            logger.info(
                "eval.batch.done",
                run_id=str(run_id),
                completed=completed,
                total=len(samples),
            )

        duration = round(time.monotonic() - start, 2)

        # Aggregate metrics
        from evalforge.metrics.base import BatchMetricResult
        good = [r for r in all_results if not r.error]
        errors = [r for r in all_results if r.error]
        aggregate = {}
        for metric in metrics:
            scores = [r.metrics.get(metric.name, 0.0) for r in good]
            r = BatchMetricResult.from_scores(metric.name, scores)
            aggregate[metric.name] = {"mean": r.mean, "std": r.std, "min": r.min, "max": r.max, "n": r.n}

        valid_latencies = [r.latency_ms for r in good]
        avg_latency = sum(valid_latencies) / len(valid_latencies) if valid_latencies else 0.0
        total_cost = sum(r.cost_usd for r in all_results)

        async with get_db_session() as db:
            run = await db.get(EvaluationRunORM, run_id)
            run.status = "done"
            run.num_samples = len(all_results)
            run.num_errors = len(errors)
            run.error_rate = len(errors) / len(all_results) if all_results else 0.0
            run.avg_latency_ms = avg_latency
            run.total_cost_usd = total_cost
            run.duration_seconds = duration
            run.aggregate_metrics = aggregate
            run.sample_results = [vars(r) for r in all_results]
            run.completed_at = datetime.now(timezone.utc)
            run.config = {**(run.config or {}), "completed_samples": len(all_results)}
            flag_modified(run, "config")
            await db.flush()
            await save_leaderboard_csv(run, task_type, dataset_name, db)

        logger.info(
            "eval.task.done",
            run_id=str(run_id),
            samples=len(all_results),
            errors=len(errors),
            duration=duration,
        )

    except Exception as exc:
        logger.error("eval.task.failed", run_id=str(run_id), error=str(exc))
        async with get_db_session() as db:
            run = await db.get(EvaluationRunORM, run_id)
            if run:
                run.status = "error"
                run.error_message = str(exc)
                run.completed_at = datetime.now(timezone.utc)


async def launch_run(db: AsyncSession, body: EvalRunCreate) -> EvaluationRunORM:
    """
    Validate inputs, create the run record, fire a background task, return immediately.
    The caller receives the run with status='running' — poll GET /evaluations/{id} for progress.
    """
    from app.api.v1.endpoints.models import MODEL_CATALOG

    dataset_row = await db.get(DatasetORM, body.dataset_id)
    if not dataset_row:
        raise NotFoundError(f"Dataset {body.dataset_id} not found")
    if dataset_row.status != "ready":
        raise DatasetError(
            f"Dataset not ready (status: {dataset_row.status})"
            + (f". Error: {dataset_row.error_message}" if dataset_row.error_message else "")
        )

    entry = MODEL_CATALOG.get(body.model_id)
    if not entry:
        raise NotFoundError(f"Model '{body.model_id}' not found in registry")
    if entry.available is False:
        raise ProviderError(
            f"Model '{body.model_id}' failed connectivity check. "
            "Run POST /models/ping-all to refresh availability."
        )

    hf_id = getattr(dataset_row, "hf_dataset_id", None) or ""
    dataset_key = (hf_id or dataset_row.name or "").lower()
    chosen_metrics = (
        body.metrics
        or _DATASET_DEFAULT_METRIC.get(dataset_key)
        or _TASK_DEFAULT_METRIC.get(dataset_row.task_type, ["answer_match"])
    )
    effective_concurrency = _provider_concurrency(entry.provider, body.concurrency)

    run = EvaluationRunORM(
        dataset_id=body.dataset_id,
        model_id=body.model_id,
        provider=entry.provider,
        status="running",
        config={
            "temperature": body.temperature,
            "max_tokens": body.max_tokens,
            "concurrency": effective_concurrency,
            "metrics": chosen_metrics,
            "num_samples": body.num_samples,
            "completed_samples": 0,
            "total_samples": None,
        },
    )
    db.add(run)
    await db.flush()
    await db.commit()

    # Fire-and-forget — background task owns its own DB sessions
    asyncio.create_task(
        _run_eval_task(
            run_id=run.id,
            dataset_id=body.dataset_id,
            chosen_metrics=chosen_metrics,
            provider_name=entry.provider,
            provider_model_id=entry.provider_model_id,
            concurrency=effective_concurrency,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
            n_samples=body.num_samples,
        ),
        name=f"eval-{run.id}",
    )

    logger.info("eval.launched", run_id=str(run.id), model=body.model_id)
    return run


async def launch_batch(db: AsyncSession, body_list: list[EvalRunCreate]) -> list[EvaluationRunORM]:
    """Launch evaluations for multiple models against the same dataset in parallel."""
    runs = []
    for body in body_list:
        run = await launch_run(db, body)
        runs.append(run)
    return runs


# Keep create_run as an alias for backward compat with benchmark endpoint
async def create_run(db: AsyncSession, body: EvalRunCreate) -> EvaluationRunORM:
    return await launch_run(db, body)

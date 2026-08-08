import uuid, statistics
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.experiment import ExperimentORM, ExperimentRunORM
from app.models.evaluation import EvaluationRunORM
from app.core.exceptions import NotFoundError

async def create_experiment(db, body) -> ExperimentORM:
    experiment = ExperimentORM(**body.model_dump())
    db.add(experiment)
    await db.flush()
    return experiment

async def list_experiments(db) -> list[ExperimentORM]:
    rows = (await db.execute(select(ExperimentORM).order_by(ExperimentORM.created_at.desc()))).scalars().all()
    return list(rows)

async def add_run(db, experiment_id: uuid.UUID, run_id: uuid.UUID) -> None:
    experiment = await db.get(ExperimentORM, experiment_id)
    
    if not experiment:
        raise NotFoundError(f"Experiment {experiment_id} not found")
    
    link = ExperimentRunORM(experiment_id=experiment_id, run_id=run_id)
    db.add(link)
    await db.flush()
    
async def get_leaderboard(db, experiment_id: uuid.UUID) -> dict:
    experiment = await db.get(ExperimentORM, experiment_id)
    
    if not experiment:
        raise NotFoundError(f"Experiment {experiment_id} not found")
    
    links = (await db.execute(select(ExperimentRunORM).where(ExperimentRunORM.experiment_id == experiment_id))).scalars().all()
    run_ids = [link.run_id for link in links]
    
    if not run_ids:
        return {
            "experiment_id": experiment_id,
            "experiment_name": experiment.name,
            "primary_metric": experiment.primary_metric,
            "entries": [],
            "total_models": 0
        }
        
    runs = (await db.execute(select(EvaluationRunORM).where(EvaluationRunORM.id.in_(run_ids), EvaluationRunORM.status == "done"))).scalars().all()
    
    # Group by model_id
    model_runs: dict[str, list] = {}
    
    for run in runs:
        model_runs.setdefault(run.model_id, []).append(run)
        
    entries = []

    for model_id, mruns in model_runs.items():
        metrics_agg: dict[str, list[float]] = {}
        latencies, costs = [], []
        total_samples = 0
        last_eval = None

        for run in mruns:
            if run.num_samples:
                total_samples += run.num_samples

            if run.avg_latency_ms:
                latencies.append(run.avg_latency_ms)

            if run.total_cost_usd and run.num_samples:
                costs.append(run.total_cost_usd / run.num_samples)

            if run.completed_at:
                if not last_eval or run.completed_at > last_eval:
                    last_eval = run.completed_at

            if run.aggregate_metrics:
                for k, v in run.aggregate_metrics.items():
                    metrics_agg.setdefault(k, []).append(v["mean"])

        entries.append({
            "model_id": model_id,
            "provider": mruns[0].provider,
            "num_runs": len(mruns),
            "total_samples": total_samples,
            "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else None,
            "avg_cost_per_sample": round(statistics.mean(costs), 6) if costs else None,
            "metrics": {
                k: {
                    "mean": round(statistics.mean(v), 4),
                    "std": round(statistics.stdev(v) if len(v) > 1 else 0.0, 4),
                    "min": round(min(v), 4),
                    "max": round(max(v), 4),
                    "num_runs": len(v)
                }
                for k, v in metrics_agg.items()
            },
            "last_evaluated": last_eval
        })

    pm = experiment.primary_metric
    entries.sort(key=lambda e: e["metrics"].get(pm, {}).get("mean", -1), reverse=True)

    for i, e in enumerate(entries):
        e["rank"] = i + 1

    return {
        "experiment_id": experiment_id,
        "experiment_name": experiment.name,
        "primary_metric": pm,
        "entries": entries,
        "total_models": len(entries)
    }


async def get_experiment_runs(db, experiment_id: uuid.UUID) -> list[EvaluationRunORM]:
    experiment = await db.get(ExperimentORM, experiment_id)

    if not experiment:
        raise NotFoundError(f"Experiment {experiment_id} not found")

    links = (await db.execute(
        select(ExperimentRunORM).where(ExperimentRunORM.experiment_id == experiment_id)
    )).scalars().all()

    if not links:
        return []

    run_ids = [link.run_id for link in links]
    runs = (await db.execute(
        select(EvaluationRunORM)
        .where(EvaluationRunORM.id.in_(run_ids))
        .order_by(EvaluationRunORM.created_at.desc())
    )).scalars().all()

    return list(runs)


async def compare_runs(db, run_ids: list[uuid.UUID]) -> dict:
    runs = []

    for rid in run_ids:
        run = await db.get(EvaluationRunORM, rid)

        if run and run.status == "done":
            runs.append(run)

    if len(runs) < 2:
        raise ValueError("Need at least 2 completed runs to compare")

    model_ids = [r.model_id for r in runs]
    aggregate = {
        r.model_id: {k: v["mean"] for k, v in (r.aggregate_metrics or {}).items()}
        for r in runs
    }

    # Match samples across runs by sample_id
    sample_map: dict[str, dict] = {}

    for run in runs:
        for s in (run.sample_results or []):
            sample_map.setdefault(s["sample_id"], {})[run.model_id] = s

    all_metrics = list({m for agg in aggregate.values() for m in agg})
    primary = all_metrics[0] if all_metrics else None

    win_counts = {m: 0 for m in model_ids}
    sample_comparisons = []

    for sid, model_samples in sample_map.items():
        ref = next(iter(model_samples.values()))
        predictions = {mid: model_samples[mid]["prediction"] for mid in model_ids if mid in model_samples}
        scores = {mid: model_samples[mid].get("metrics", {}) for mid in model_ids if mid in model_samples}

        winner = None

        if primary and scores:
            winner = max(scores, key=lambda m: scores[m].get(primary, 0))
            if winner:
                win_counts[winner] += 1

        sample_comparisons.append({
            "sample_id": sid,
            "input": ref["input"],
            "reference": ref.get("reference"),
            "predictions": predictions,
            "scores": scores,
            "winner": winner,
        })

    overall_winner = max(win_counts, key=win_counts.get) if any(win_counts.values()) else None

    return {
        "run_ids": run_ids,
        "model_ids": model_ids,
        "metrics": all_metrics,
        "aggregate_metrics": aggregate,
        "win_counts": win_counts,
        "overall_winner": overall_winner,
        "sample_comparisons": sample_comparisons,
    }
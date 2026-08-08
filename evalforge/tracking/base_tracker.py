"""
Experiment tracking abstraction.

CONCEPT: Why Track Experiments?
─────────────────────────────────
Running evals is cheap. Reproducing results six months later is hard —
unless you tracked: which model, which dataset, which prompt, which metrics,
and what the exact scores were.

EvalForge uses MLflow as the primary tracking backend (Sprint 5), with a
LocalTracker for offline development. Both implement BaseTracker so the
evaluator doesn't care which backend is active.

MLflow concepts mapped to EvalForge concepts:
  MLflow Experiment  →  An EvalForge Experiment (e.g., "MMLU Benchmark Suite")
  MLflow Run         →  One EvalReport (one model × one dataset × one config)
  MLflow Metrics     →  mean ROUGE-L, mean exact_match, avg_latency_ms, ...
  MLflow Params      →  model, provider, temperature, n_samples, ...
  MLflow Artifacts   →  per_sample_results.json, eval_config.yaml

Usage:
    tracker = MLflowTracker(experiment_name="mmlu-baseline")
    with tracker.start_run(run_name="gpt-4o-anatomy") as run_id:
        report = await evaluator.run(dataset)
        tracker.log_report(report)
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Generator

from evalforge.evaluators.base_evaluator import EvalReport


class BaseTracker(ABC):
    """Abstract interface for experiment tracking backends."""

    @abstractmethod
    def start_run(self, run_name: str, tags: dict[str, str] | None = None) -> str:
        """
        Start a new tracking run. Returns a run_id string.
        Implementations should return a context manager if the backend
        requires explicit end_run() calls.
        """

    @abstractmethod
    def end_run(self) -> None:
        """End the current active run."""

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters / configuration for the active run."""

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log numeric metrics for the active run."""

    @abstractmethod
    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Upload a local file as a run artifact."""

    def log_report(self, report: EvalReport) -> None:
        """
        Convenience method: log a complete EvalReport to the active run.

        Logs params (model, dataset, config) + all aggregate metrics.
        """
        self.log_params({
            "model": report.model,
            "provider": report.provider,
            "dataset": report.dataset_name,
            "split": report.split,
            "n_samples": report.n_samples,
            "n_errors": report.n_errors,
            **{f"config.{k}": str(v) for k, v in report.config.items()},
        })
        self.log_metrics({
            **{name: result.mean for name, result in report.aggregate_metrics.items()},
            "error_rate": report.error_rate,
            "total_cost_usd": report.total_cost_usd,
            "avg_latency_ms": report.avg_latency_ms,
            "duration_seconds": report.duration_seconds,
        })


class LocalTracker(BaseTracker):
    """
    File-based tracker for offline development.

    Logs params and metrics to the console and saves reports as JSON.
    Use this when MLflow is not running.
    """

    def __init__(self) -> None:
        self._active_run_id: str | None = None

    def start_run(self, run_name: str, tags: dict[str, str] | None = None) -> str:
        import uuid
        self._active_run_id = str(uuid.uuid4())
        print(f"[LocalTracker] Starting run: {run_name} ({self._active_run_id})")
        return self._active_run_id

    def end_run(self) -> None:
        print(f"[LocalTracker] Ending run: {self._active_run_id}")
        self._active_run_id = None

    def log_params(self, params: dict[str, Any]) -> None:
        for k, v in params.items():
            print(f"[LocalTracker] PARAM  {k} = {v}")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        for k, v in metrics.items():
            print(f"[LocalTracker] METRIC {k} = {v:.4f}")

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        print(f"[LocalTracker] ARTIFACT {local_path}")

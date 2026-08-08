"""
Base evaluation orchestrator — the core evaluation loop.

CONCEPT: The Evaluation Loop
──────────────────────────────
Every evaluation run follows the same five steps:

  ┌────────────────────────────────────────────────────────────┐
  │  1. Load dataset          → list[EvalSample]               │
  │  2. Build prompt          → GenerationRequest              │
  │  3. Call model            → GenerationResponse             │
  │  4. Extract answer        → str (normalized prediction)    │
  │  5. Apply metrics         → MetricResult per metric        │
  └────────────────────────────────────────────────────────────┘
  Then aggregate all sample results → EvalReport

BaseEvaluator handles steps 3–5 + aggregation + concurrency.
Subclasses override:
  - build_prompt()    → step 2 (task-specific formatting)
  - extract_answer()  → step 4 (parsing, e.g. "A" from "The answer is A")

CONCEPT: Concurrent Evaluation with Semaphore
───────────────────────────────────────────────
Evaluating 1000 samples sequentially at 500ms/call = 8 minutes.
With concurrency=10: 1000 samples / 10 parallel = ~50 seconds.

asyncio.Semaphore(concurrency) ensures at most N calls run simultaneously,
preventing rate-limit errors while maximizing throughput.

CONCEPT: Exponential Backoff on Retries
─────────────────────────────────────────
Provider APIs are flaky. A single 429 (rate limit) shouldn't fail the run.
We retry with exponential backoff: 1s, 2s, 4s between attempts.
After max_retries, the sample is marked as error (not a crash).
"""
import asyncio, time, uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from app.core.logging import get_logger
from evalforge.datasets.base import BaseDataset, EvalSample
from evalforge.metrics.base import BaseMetric, BatchMetricResult
from evalforge.providers.base import BaseProvider, GenerationRequest, GenerationResponse

logger = get_logger(__name__)

@dataclass
class SampleResult:
    """
    Evaluation result for a single sample.

    Fields mirror what gets stored in the DB (Sprint 2+) and logged to MLflow.
    """
    sample_id: str
    input: str
    reference: str | None
    prediction: str
    metrics: dict[str, float]       # metric_name → score
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    error: str | None = None        # Set if provider call failed after all retries

@dataclass
class EvalReport:
    """
    Aggregated results for a complete evaluation run.

    This is the top-level return value of BaseEvaluator.run().
    It contains:
      - Aggregate statistics per metric (mean, std, min, max)
      - Per-sample details for error analysis
      - Timing, cost, and efficiency stats

    The summary() method returns a compact dict suitable for
    logging to MLflow and returning from the API.
    """
    run_id: str
    dataset_name: str
    model: str
    provider: str
    split: str
    num_samples: int
    n_errors: int
    duration_seconds: float
    aggregate_metrics: dict[str, BatchMetricResult]
    sample_results: list[SampleResult]
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def error_rate(self) -> float:
        return self.n_errors / self.num_samples if self.num_samples else 0.0

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.sample_results)

    @property
    def avg_latency_ms(self) -> float:
        valid = [r.latency_ms for r in self.sample_results if not r.error]
        return sum(valid) / len(valid) if valid else 0.0

    def summary(self) -> dict[str, Any]:
        """Compact dict for MLflow logging and API responses."""
        return {
            "run_id": self.run_id,
            "dataset": self.dataset_name,
            "model": self.model,
            "provider": self.provider,
            "num_samples": self.num_samples,
            "error_rate": round(self.error_rate, 4),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "duration_seconds": self.duration_seconds,
            "metrics": {
                name: round(result.mean, 4)
                for name, result in self.aggregate_metrics.items()
            },
        }

class BaseEvaluator(ABC):
    """
    Abstract base for all EvalForge evaluation pipelines.

    Concrete evaluators:
      - BenchmarkEvaluator  (Sprint 4) — standard benchmarks (MMLU, GSM8K)
      - CustomEvaluator     (Sprint 4) — custom CSV/JSONL datasets
      - ComparisonEvaluator (Sprint 4) — pairwise A/B model comparison
      - LLMJudgeEvaluator   (Sprint 6) — GPT-4/Claude as evaluator
      - RAGEvaluator        (Sprint 7) — retrieval + generation pipeline
      - SQLEvaluator        (Sprint 8) — SQL correctness evaluation

    The base class provides:
      - Concurrent async execution with semaphore
      - Per-sample retry with exponential backoff
      - Result aggregation
    """

    def __init__(
        self,
        provider: BaseProvider,
        metrics: list[BaseMetric],
        concurrency: int = 10,
        max_retries: int = 3,
        max_tokens: int = 512,
        temperature: float = 0
    ) -> None:
        self.provider = provider
        self.metrics = metrics
        self.concurrency = concurrency
        self.max_retries = max_retries
        self._semaphore = asyncio.Semaphore(concurrency)
        self._max_tokens = max_tokens
        self._temperature = temperature

    @abstractmethod
    def build_prompt(self, sample: EvalSample) -> GenerationRequest:
        """
        Convert an EvalSample into a GenerationRequest.

        This is where task-specific prompt engineering lives.

        Examples:
          - BenchmarkEvaluator: formats multiple-choice with A/B/C/D options
          - CustomEvaluator: fills in a user-defined prompt template
          - SQLEvaluator: adds schema context + few-shot SQL examples
        """

    @abstractmethod
    def extract_answer(self, response: GenerationResponse, sample: EvalSample) -> str:
        """
        Parse the model's raw output into a normalized answer string.

        Why this matters:
          A model might output "The answer is A. Paris is the capital..."
          but the metric expects just "A". extract_answer() does that parsing.

        Examples:
          - Multiple-choice: extract first letter "A", "B", "C", or "D"
          - Numeric: extract the first number from the response
          - Open-ended: strip and return response.text as-is
        """

    # Error substrings that indicate a permanent failure — no point retrying
    _FATAL_ERROR_PATTERNS = (
        "credit balance is too low",
        "insufficient_funds",
        "billing",
        "Your account is not active",
        "account has been deactivated",
        "invalid api key",
        "incorrect api key",
        "authentication",
    )

    @staticmethod
    def _is_fatal(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(p in msg for p in BaseEvaluator._FATAL_ERROR_PATTERNS)

    async def _evaluate_sample_with_retry(self, sample: EvalSample) -> SampleResult:
        """ Evaluates one sample and retries on transient failures; skips retries on fatal errors. """
        async with self._semaphore:
            last_error: Exception | None = None

            for attempt in range(self.max_retries):
                try:
                    request = self.build_prompt(sample)
                    response = await self.provider.generate(request)
                    prediction = self.extract_answer(response, sample)

                    metric_scores: dict[str, float] = {}
                    if sample.reference is not None:
                        for metric in self.metrics:
                            result = metric.score(
                                prediction, sample.reference,
                                choices=sample.choices,
                                label=sample.label,
                            )
                            metric_scores[metric.name] = result.score

                    return SampleResult(
                        sample_id=sample.id,
                        input=sample.input,
                        reference=sample.reference,
                        prediction=prediction,
                        metrics=metric_scores,
                        latency_ms=response.latency_ms,
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens,
                        cost_usd=self.provider.estimate_cost(response),
                    )

                except Exception as exc:
                    last_error = exc
                    if self._is_fatal(exc):
                        logger.error(
                            "evaluator.fatal_error",
                            sample_id=sample.id,
                            error=str(exc)[:200],
                        )
                        break  # no retry on billing/auth failures
                    wait = 2 ** attempt
                    logger.warning(
                        "evaluator.retry",
                        sample_id=sample.id,
                        attempt=attempt + 1,
                        wait_seconds=wait,
                        error=str(exc),
                    )
                    await asyncio.sleep(wait)

            logger.error(
                "evaluator.sample.failed",
                sample_id=sample.id,
                error=str(last_error),
            )
            return SampleResult(
                sample_id=sample.id,
                input=sample.input,
                reference=sample.reference,
                prediction="",
                metrics={},
                latency_ms=0.0,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                error=str(last_error),
            )

    async def run(self, dataset: BaseDataset, run_id: str | None = None, num_samples: int | None = None) -> EvalReport:
        """
        Run evaluation across the dataset and return an EvalReport.

        Args:
            dataset:    The dataset to evaluate on.
            run_id:     Optional unique run identifier (auto-generated if None).
            num_samples:  Evaluate only the first N samples (useful for quick pilots).

        Returns:
            EvalReport with per-metric aggregate stats and per-sample details.
        """
        run_id = run_id or str(uuid.uuid4())
        samples = dataset.samples[:num_samples] if num_samples else dataset.samples

        logger.info(
            "evaluator.run.start",
            run_id=run_id,
            dataset=dataset.info.name,
            num_samples=len(samples),
            provider=self.provider.provider_name,
            model=self.provider.model,
            concurrency=self.concurrency,
        )

        start = time.monotonic()
        results: list[SampleResult] = await asyncio.gather(
            *[self._evaluate_sample_with_retry(s) for s in samples]
        )
        duration = round(time.monotonic() - start, 2)

        # Aggregate metrics across non-error samples
        good = [r for r in results if not r.error]
        errors = [r for r in results if r.error]

        aggregate: dict[str, BatchMetricResult] = {}
        for metric in self.metrics:
            scores = [r.metrics.get(metric.name, 0.0) for r in good]
            aggregate[metric.name] = BatchMetricResult.from_scores(metric.name, scores)

        report = EvalReport(
            run_id=run_id,
            dataset_name=dataset.info.name,
            model=self.provider.model,
            provider=self.provider.provider_name,
            split=dataset.split,
            num_samples=len(samples),
            n_errors=len(errors),
            duration_seconds=duration,
            aggregate_metrics=aggregate,
            sample_results=list(results),
        )

        logger.info("evaluator.run.complete", **report.summary())
        return report
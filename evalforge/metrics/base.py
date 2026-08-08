"""
Base class for all evaluation metrics.

CONCEPT: Metric Families in LLM Evaluation
────────────────────────────────────────────
After collecting model outputs, we measure quality using four metric families:

  ┌──────────────────┬────────────────────────────────────────────────┐
  │ Family           │ Examples              │ When to use            │
  ├──────────────────┼────────────────────────────────────────────────┤
  │ 1. Text Overlap  │ BLEU, ROUGE, ExactMatch│ Fixed-answer tasks    │
  │                  │                        │ SQL gen, QA, code     │
  ├──────────────────┼────────────────────────────────────────────────┤
  │ 2. Semantic      │ BERTScore, CosineSim   │ Paraphrase-tolerant   │
  │                  │                        │ Summarization, transl.│
  ├──────────────────┼────────────────────────────────────────────────┤
  │ 3. LLM-as-Judge  │ GPT-4 rubric scoring   │ Open-ended generation │
  │                  │ Pairwise preference    │ Instruction following │
  ├──────────────────┼────────────────────────────────────────────────┤
  │ 4. Production    │ Latency, Cost, Safety  │ Always — alongside    │
  │                  │ Token usage            │ quality metrics       │
  └──────────────────┴────────────────────────────────────────────────┘

CONCEPT: Aggregate vs Per-Sample Results
──────────────────────────────────────────
score()        → MetricResult      one (prediction, reference) pair
score_batch()  → BatchMetricResult the whole dataset, with mean/std/min/max

Always report both. Per-sample results let you identify failure modes
(e.g., "model fails on questions longer than 200 tokens").
Aggregate tells you the headline number to compare across models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

@dataclass
class MetricResult:
    """
    Result for a single (prediction, reference) pair.

    Attributes:
        name:     Metric identifier, e.g. "exact_match", "rouge_l".
        score:    Numeric score (range depends on metric — see score_range).
        details:  Metric-specific breakdown dict, e.g.:
                  {"rouge1": 0.8, "rouge2": 0.6, "rougeL": 0.75} for ROUGE.
        passed:   Whether score >= threshold (set only if threshold is given).
    """
    name: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)
    passed: bool | None = None

@dataclass
class BatchMetricResult:
    """
    Aggregated metric results across a full dataset or evaluation run.

    This is what gets logged to MLflow and shown in the leaderboard.

    Attributes:
        name:       Metric name.
        mean:       Average score — the headline number.
        std:        Standard deviation — spread across samples.
        min / max:  Worst / best sample scores.
        per_sample: Individual score for each sample (same order as dataset).
        n:          Number of samples evaluated.
    """
    name: str
    mean: float
    std: float
    min: float
    max: float
    per_sample: list[float]
    n: int

    @classmethod
    def from_scores(cls, name: str, scores: list[float]) -> "BatchMetricResult":
        """Build a BatchMetricResult from a flat list of per-sample scores."""
        import statistics
        if not scores:
            return cls(name=name, mean=0.0, std=0.0, min=0.0, max=0.0, per_sample=[], n=0)
        return cls(
            name=name,
            mean=round(statistics.mean(scores), 6),
            std=round(statistics.stdev(scores) if len(scores) > 1 else 0.0, 6),
            min=round(min(scores), 6),
            max=round(max(scores), 6),
            per_sample=scores,
            n=len(scores),
        )

    def __repr__(self) -> str:
        return f"BatchMetricResult(name={self.name!r}, mean={self.mean:.4f}, n={self.n})"

class BaseMetric(ABC):
    """
    Abstract base for all EvalForge metrics.

    Contract:
      - score()        must be implemented (single pair)
      - score_batch()  has a working default (loops over score())
                       Override for vectorized implementations (BERTScore, etc.)
      - name           must be a stable lowercase string (used as dict key)

    Usage:
        metric = ExactMatchMetric()

        # Single pair
        result = metric.score("Paris", "Paris")
        print(result.score)  # 1.0

        # Batch
        batch = metric.score_batch(
            predictions=["Paris", "London", "Berlin"],
            references= ["Paris", "Berlin", "Berlin"],
        )
        print(batch.mean)        # 0.6667
        print(batch.per_sample)  # [1.0, 0.0, 1.0]
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable lowercase identifier: "exact_match", "rouge_l", "bertscore", ..."""

    @property
    def score_range(self) -> tuple[float, float]:
        """(min, max) for this metric. Default: (0.0, 1.0)."""
        return (0.0, 1.0)

    @property
    def higher_is_better(self) -> bool:
        """True for most metrics (accuracy, F1). False for perplexity, latency."""
        return True

    @abstractmethod
    def score(self, prediction: str, reference: str, **kwargs: Any) -> MetricResult:
        """Compute the metric for a single (prediction, reference) pair."""

    def score_batch(
        self,
        predictions: list[str],
        references: list[str],
        **kwargs: Any,
    ) -> BatchMetricResult:
        """
        Compute metric for a batch of (prediction, reference) pairs.

        Default: iterates score() for each pair. Override this for efficiency:
          - BERTScore: processes the whole batch in one forward pass
          - ROUGE: can batch in pure Python without GPU

        Args:
            predictions: Model outputs, one per sample.
            references:  Gold answers, aligned with predictions.
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"Lengths must match: {len(predictions)} predictions vs "
                f"{len(references)} references."
            )
        scores = [
            self.score(pred, ref, **kwargs).score
            for pred, ref in zip(predictions, references)
        ]
        return BatchMetricResult.from_scores(self.name, scores)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
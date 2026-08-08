"""
Tests for the BaseMetric contract.

These tests use a concrete ExactMatchMetric (Sprint 4) stub to validate
the base class plumbing: batch scoring, score range, aggregation math.
"""
import pytest

class _StubMetric:
    """ Minimal concrete metric for testing BaseMetric contract. """
    name = "stub"
    score_range = (0.0, 1.0)
    higher_is_better = True

    def score(self, prediction: str, reference: str, **_):
        from evalforge.metrics.base import MetricResult
        return MetricResult(name=self.name, score=1.0 if prediction == reference else 0.0)

    def score_batch(self, predictions, references, **kwargs):
        from evalforge.metrics.base import BatchMetricResult
        scores = [self.score(p, r).score for p, r in zip(predictions, references)]
        return BatchMetricResult.from_scores(self.name, scores)

def test_score_exact_match():
    metric = _StubMetric()
    assert metric.score("Paris", "Paris").score == 1.0
    assert metric.score("London", "Paris").score == 0.0

def test_score_batch_mean():
    from evalforge.metrics.base import BatchMetricResult
    metric = _StubMetric()
    result = metric.score_batch(
        predictions=["Paris", "London", "Berlin"],
        references= ["Paris", "Berlin", "Berlin"],
    )
    assert isinstance(result, BatchMetricResult)
    assert result.n == 3
    assert abs(result.mean - 2/3) < 1e-6

def test_score_batch_empty():
    from evalforge.metrics.base import BatchMetricResult
    result = BatchMetricResult.from_scores("stub", [])
    assert result.n == 0
    assert result.mean == 0.0

def test_score_batch_length_mismatch():
    metric = _StubMetric()
    with pytest.raises(ValueError, match="Lengths must match"):
        metric.score_batch(["a", "b"], ["c"])
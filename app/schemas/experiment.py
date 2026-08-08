import uuid
from datetime import datetime
from pydantic import BaseModel

class ExperimentCreate(BaseModel):
    name: str
    description: str | None = None
    dataset_id: uuid.UUID | None = None
    benchmark_name: str | None = None
    primary_metric: str = "answer_match"
    tags: list[str] | None = None
    
class ExperimentResponse(BaseModel):
    id: uuid.UUID
    name: str
    description: str | None
    dataset_id: uuid.UUID | None
    benchmark_name: str | None
    primary_metric: str
    tags: list[str] | None
    created_at: datetime
    
    model_config = {"from_attributes": True}
    
class AddRunToExperiment(BaseModel):
    run_id: uuid.UUID
    
# Leaderboard schemas
class ModelMetricSummary(BaseModel):
    mean: float
    std: float
    min: float
    max: float
    num_runs: int
    
class LeaderboardEntry(BaseModel):
    rank: int
    model_id: str
    provider: str
    num_runs: int
    total_samples: int
    avg_latency_ms: float | None
    avg_cost_per_sample: float | None
    metrics: dict[str, ModelMetricSummary]
    last_evaluated: datetime | None
    
class LeaderboardResponse(BaseModel):
    experiment_id: uuid.UUID
    experiment_name: str
    primary_metric: str
    entries: list[LeaderboardEntry]
    total_models: int
    
class SampleComparison(BaseModel):
    sample_id: str
    input: str
    reference: str | None
    predictions: dict[str, str] # model_id -> prediction
    scores: dict[str, dict[str, float]] # model_id -> {metric -> score}
    winner: str | None # model_id with best primary metric score
    
class ComparisonResponse(BaseModel):
    run_ids: list[uuid.UUID]
    model_ids: list[str]
    metrics: list[str]
    aggregate_metrics: dict[str, dict[str, float]] # metric_id -> {metric: mean}
    win_counts: dict[str, int] # model_id -> num samples won
    overall_winner: str | None
    sample_comparisons: list[SampleComparison]
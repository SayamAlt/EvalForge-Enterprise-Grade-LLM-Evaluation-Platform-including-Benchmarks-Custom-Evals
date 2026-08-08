import uuid
from datetime import datetime
from pydantic import BaseModel, Field

class EvalRunCreate(BaseModel):
    dataset_id: uuid.UUID
    model_id: str = Field(..., examples=["gpt-3.5-turbo", "gemma2-9b", "deepseek-chat"])
    num_samples: int | None = Field(default=None, description="Limit the number of samples to evaluate.")
    concurrency: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1)
    metrics: list[str] | None = Field(default=None, description="Metrics to compute. Defaults to task-appropriate metric.")
    
class AggregateMetricResult(BaseModel):
    mean: float
    std: float
    min: float
    max: float
    n: int

class SampleResultResponse(BaseModel):
    sample_id: str
    input: str
    reference: str | None
    prediction: str
    metrics: dict[str, float]
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    error: str | None

class EvalRunResponse(BaseModel):
    id: uuid.UUID
    dataset_id: uuid.UUID | None
    model_id: str
    provider: str
    status: str
    num_samples: int | None
    num_errors: int | None
    error_rate: float | None
    avg_latency_ms: float | None
    total_cost_usd: float | None
    duration_seconds: float | None
    aggregate_metrics: dict[str, AggregateMetricResult] | None
    sample_results: list[SampleResultResponse] | None
    error_message: str | None
    created_at: datetime
    completed_at: datetime | None

    model_config = {"from_attributes": True}
    
class EvalRunSummaryResponse(BaseModel):
    """Lightweight version for list endpoints — omits sample_results."""
    id: uuid.UUID
    dataset_id: uuid.UUID | None
    model_id: str
    provider: str
    status: str
    num_samples: int | None
    num_errors: int | None
    error_rate: float | None
    avg_latency_ms: float | None
    total_cost_usd: float | None
    duration_seconds: float | None
    aggregate_metrics: dict[str, AggregateMetricResult] | None
    error_message: str | None
    created_at: datetime
    completed_at: datetime | None

    model_config = {"from_attributes": True}

class EvalRunListResponse(BaseModel):
    items: list[EvalRunSummaryResponse]
    total: int
    page: int
    page_size: int


class BatchEvalCreate(BaseModel):
    dataset_id: uuid.UUID
    model_ids: list[str] = Field(..., min_length=1, description="Models to evaluate in parallel")
    num_samples: int | None = Field(default=None)
    concurrency: int = Field(default=5, ge=1, le=20)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1)
    metrics: list[str] | None = Field(default=None)
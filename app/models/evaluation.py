import uuid
from datetime import datetime
from sqlalchemy import String, Float, Integer, JSON, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from app.models.base import Base

class EvaluationRunORM(Base):
    __tablename__ = "evaluation_runs"

    dataset_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    model_id: Mapped[str] = mapped_column(String, nullable=False)
    provider: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, default="pending") # pending | running | done | error
    num_samples: Mapped[int | None] = mapped_column(Integer, nullable=True)
    num_errors: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    aggregate_metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    sample_results: Mapped[list | None] = mapped_column(JSON, nullable=True)
    benchmark_name: Mapped[str | None] = mapped_column(String, nullable=True)                                                                           
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(String, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
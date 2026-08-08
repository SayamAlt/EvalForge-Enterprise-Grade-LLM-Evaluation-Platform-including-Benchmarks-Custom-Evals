import uuid
from sqlalchemy import String, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from app.models.base import Base

class ExperimentORM(Base):
    __tablename__ = "experiments"                                                                                                                   
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str | None] = mapped_column(String, nullable=True)
    dataset_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)                                                           
    benchmark_name: Mapped[str | None] = mapped_column(String, nullable=True)
    primary_metric: Mapped[str] = mapped_column(String, default="answer_match")                                                                       
    tags: Mapped[list | None] = mapped_column(JSON, nullable=True)

class ExperimentRunORM(Base):
    __tablename__ = "experiment_runs"                                                                                                                 
    # Override base id/created_at/updated_at - this is a join table                                                                                 
    __abstract__ = False                                                                                                                              
    # Actually just add FKs as plain columns
    experiment_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)                                                              
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
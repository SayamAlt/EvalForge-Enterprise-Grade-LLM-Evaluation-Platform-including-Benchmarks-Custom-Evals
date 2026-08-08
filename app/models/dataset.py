"""
SQLAlchemy ORM model for datasets.

CONCEPT: What we persist vs what we compute on demand
──────────────────────────────────────────────────────
We store metadata (name, source, column mappings), NOT the actual samples.

Why? A 14,000-row MMLU dataset would be wasteful to store as JSON blobs in
Postgres — we'd pay serialization cost on every read with zero query benefit.
Instead, the DB stores enough to reconstruct EvalSamples when needed:
  - File uploads: file_path → read from disk
  - HuggingFace: hf_dataset_id + config + split → re-fetch from HF Hub cache

Status transitions:
  "pending" → "ready"   (parse succeeded, num_samples set)
  "pending" → "error"   (parse failed, error_message set)

The format_config field bridges arbitrary file schemas to EvalSample fields.
Example for a file with non-standard columns:
  {"input_column": "my_question", "reference_column": "gold_answer"}
"""
from sqlalchemy import Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from app.models.base import Base

class DatasetORM(Base):
    __tablename__ = "datasets"

    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # "csv" | "jsonl" | "huggingface"
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)

    # "multiple_choice" | "generation" | "qa" | "code" | "classification"
    task_type: Mapped[str] = mapped_column(String(50), nullable=False)

    # "pending" | "ready" | "error"
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending")

    # Set after successful parse — None until then
    num_samples: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # For file uploads: path to stored file on disk (or object storage key in prod)
    file_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # For HuggingFace Hub datasets
    hf_dataset_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    hf_config: Mapped[str | None] = mapped_column(String(100), nullable=True)
    hf_split: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Column mapping: {"input_column": "question", "reference_column": "answer", ...}
    # None means use auto-detection
    format_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Populated only when status == "error"
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
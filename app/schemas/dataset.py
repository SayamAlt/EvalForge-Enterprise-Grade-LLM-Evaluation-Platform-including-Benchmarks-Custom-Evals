"""
Pydantic schemas for the Dataset API.

CONCEPT: Why schemas are separate from ORM models
──────────────────────────────────────────────────
ORM models map to DB columns. Schemas map to API contracts. They look similar
but serve different purposes:

  DatasetORM   → SQLAlchemy, knows about DB, has relationships
  DatasetRead  → Pydantic, knows about JSON, has validators

Decoupling lets you:
  - Expose different fields than what's in the DB (e.g. hide file_path)
  - Rename fields for the API without touching the DB schema
  - Add computed fields that don't exist in the DB

Two create schemas because file upload and HuggingFace registration need
completely different required fields:
  DatasetFileCreate → name, task_type, optional column overrides
  DatasetHFCreate   → name, task_type, hf_dataset_id, hf_config, hf_split
"""
import uuid
from datetime import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict, Field

TASK_TYPE_PATTERN = "^(multiple_choice|generation|qa|code|classification)$"

class DatasetFileCreate(BaseModel):
    """ Metadata accompanying a file upload (sent as multipart form fields). """
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    task_type: str = Field(..., pattern=TASK_TYPE_PATTERN)
    # Optional column name overrides; if None, auto-detect from column names
    format_config: dict[str, str] | None = None

class DatasetHFCreate(BaseModel):
    """ Registers a HuggingFace Hub dataset without uploading a file. """
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    task_type: str = Field(..., pattern=TASK_TYPE_PATTERN)
    hf_dataset_id: str = Field(..., description="HuggingFace dataset repo ID, e.g. 'cais/mmlu'")
    hf_config: str | None = Field(None, description="Dataset config/subset name, e.g. 'anatomy' for MMLU")
    hf_split: str = Field("test", description="Dataset split: 'train', 'test', or 'validation'")
    format_config: dict[str, str] | None = None

class DatasetRead(BaseModel):
    """ API response for a single dataset. Does not expose file_path (internal). """
    id: uuid.UUID
    name: str
    description: str | None
    source_type: str
    task_type: str
    status: str
    num_samples: int | None
    hf_dataset_id: str | None
    hf_config: str | None
    hf_split: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    model_config = ConfigDict(from_attributes=True)

class DatasetListResponse(BaseModel):
    items: list[DatasetRead]
    total: int
    page: int
    page_size: int

class SamplePreview(BaseModel):
    """ A single EvalSample rendered as JSON for the preview endpoint. """
    id: str
    input: str
    reference: str | None
    choices: list[str] | None
    label: int | None
    metadata: dict[str, Any]

class DatasetPreviewResponse(BaseModel):
    dataset_id: uuid.UUID
    total_samples: int
    preview_count: int
    samples: list[SamplePreview]
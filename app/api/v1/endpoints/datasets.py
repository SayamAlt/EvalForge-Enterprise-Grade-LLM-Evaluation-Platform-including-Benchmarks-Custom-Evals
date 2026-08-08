"""
Dataset REST endpoints.

    POST   /api/v1/datasets/upload          Upload CSV or JSONL file
    POST   /api/v1/datasets/huggingface     Register HuggingFace Hub dataset
    GET    /api/v1/datasets                 List datasets (paginated)
    GET    /api/v1/datasets/{id}            Get dataset details
    GET    /api/v1/datasets/{id}/preview    Show first N samples
    DELETE /api/v1/datasets/{id}            Delete dataset

CONCEPT: Multipart form for file uploads
─────────────────────────────────────────
JSON body doesn't support binary file uploads. For file uploads we use
multipart/form-data: the file bytes are one "part" and metadata fields are
separate string "parts". FastAPI handles this with File() and Form().

This is why upload_dataset uses Form() for name/task_type instead of a
Pydantic body — you can't mix File() and a JSON body in the same request.
The HuggingFace endpoint (no file) uses a regular JSON body.
"""

import uuid

from fastapi import APIRouter, File, Form, Query, UploadFile, status

from app.api.deps import DbSession
from app.schemas.dataset import (
    DatasetHFCreate,
    DatasetListResponse,
    DatasetPreviewResponse,
    DatasetRead,
    SamplePreview,
)
from app.services import dataset_service

router = APIRouter()


@router.post(
    "/upload",
    response_model=DatasetRead,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a CSV or JSONL dataset",
)
async def upload_dataset(
    db: DbSession,
    file: UploadFile = File(..., description="CSV or JSONL evaluation dataset"),
    name: str = Form(..., min_length=1),
    task_type: str = Form(..., description="multiple_choice | generation | qa | code | classification"),
    description: str | None = Form(None),
    input_column: str | None = Form(None, description="CSV column for model input"),
    reference_column: str | None = Form(None, description="CSV column for expected output"),
    choices_column: str | None = Form(None, description="CSV column for answer choices (MC tasks)"),
    label_column: str | None = Form(None, description="CSV column for correct answer index (MC tasks)"),
) -> DatasetRead:
    """
    Upload a CSV or JSONL file as an evaluation dataset.

    Column auto-detection maps common names automatically:
    - input: `question`, `prompt`, `text`, `input`, `instruction`
    - reference: `answer`, `output`, `target`, `reference`, `label`
    - choices: `choices`, `options`, `candidates`
    - label: `label`, `answer_index`, `correct_index`

    If auto-detection fails, provide explicit column names via the form fields.

    **Example CSV:**
    ```
    question,answer
    "What is 2+2?","4"
    "Capital of France?","Paris"
    ```

    **Example JSONL:**
    ```
    {"question": "What is 2+2?", "answer": "4"}
    {"question": "Capital of France?", "answer": "Paris"}
    ```
    """
    from app.schemas.dataset import DatasetFileCreate

    format_config = {
        k: v for k, v in {
            "input_column": input_column,
            "reference_column": reference_column,
            "choices_column": choices_column,
            "label_column": label_column,
        }.items() if v is not None
    } or None

    metadata = DatasetFileCreate(
        name=name,
        description=description,
        task_type=task_type,
        format_config=format_config,
    )

    dataset = await dataset_service.create_from_file(db, file, metadata)
    return DatasetRead.model_validate(dataset)


@router.post(
    "/huggingface",
    response_model=DatasetRead,
    status_code=status.HTTP_201_CREATED,
    summary="Register a HuggingFace Hub dataset",
)
async def register_hf_dataset(
    db: DbSession,
    body: DatasetHFCreate,
) -> DatasetRead:
    """
    Register a dataset from HuggingFace Hub without uploading a file.

    The dataset is downloaded and cached on first use (~10-60s for large datasets).
    Subsequent registrations of the same dataset use the local cache.

    **Supported benchmarks with automatic schema detection:**
    - `cais/mmlu` — configs: `"all"`, `"anatomy"`, `"abstract_algebra"`, ...
    - `gsm8k` — config: `"main"`
    - `Rowan/hellaswag`
    - `truthful_qa` — config: `"multiple_choice"`
    - `openai/openai_humaneval`

    **Example body:**
    ```json
    {
      "name": "MMLU Anatomy Test",
      "task_type": "multiple_choice",
      "hf_dataset_id": "cais/mmlu",
      "hf_config": "anatomy",
      "hf_split": "test"
    }
    ```
    """
    dataset = await dataset_service.create_from_huggingface(db, body)
    return DatasetRead.model_validate(dataset)


@router.get(
    "",
    response_model=DatasetListResponse,
    summary="List all registered datasets",
)
async def list_datasets(
    db: DbSession,
    page: int = Query(1, ge=1, description="Page number (1-based)"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
) -> DatasetListResponse:
    """Return a paginated list of datasets, sorted newest first."""
    items, total = await dataset_service.list_datasets(db, page, page_size)
    return DatasetListResponse(
        items=[DatasetRead.model_validate(d) for d in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get(
    "/{dataset_id}",
    response_model=DatasetRead,
    summary="Get dataset metadata",
)
async def get_dataset(
    db: DbSession,
    dataset_id: uuid.UUID,
) -> DatasetRead:
    """Fetch metadata for a single dataset by its UUID."""
    dataset = await dataset_service.get_dataset(db, dataset_id)
    return DatasetRead.model_validate(dataset)


@router.get(
    "/{dataset_id}/preview",
    response_model=DatasetPreviewResponse,
    summary="Preview first N samples from a dataset",
)
async def preview_dataset(
    db: DbSession,
    dataset_id: uuid.UUID,
    n: int = Query(5, ge=1, le=50, description="Number of samples to preview"),
) -> DatasetPreviewResponse:
    """
    Return the first N samples as EvalSample JSON.

    Use this to verify column mapping was correct before running a full evaluation.
    The response shows exactly what the evaluation engine will see.
    """
    dataset, samples = await dataset_service.get_preview(db, dataset_id, n)
    return DatasetPreviewResponse(
        dataset_id=dataset.id,
        total_samples=dataset.num_samples or 0,
        preview_count=len(samples),
        samples=[
            SamplePreview(
                id=s.id,
                input=s.input,
                reference=s.reference,
                choices=s.choices,
                label=s.label,
                metadata=s.metadata,
            )
            for s in samples
        ],
    )


@router.delete(
    "/{dataset_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a dataset",
)
async def delete_dataset(
    db: DbSession,
    dataset_id: uuid.UUID,
) -> None:
    """
    Permanently delete a dataset and its uploaded file.
    Evaluation results that referenced this dataset are preserved.
    """
    await dataset_service.delete_dataset(db, dataset_id)

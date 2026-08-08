"""
Dataset service — business logic for all dataset operations.

CONCEPT: Service layer role

Routes handle HTTP: validate input, call service, serialize response.
Services handle business logic: talk to DB, call loaders, manage files.

This separation means:
  - Routes stay < 10 lines each
  - Services are testable without HTTP overhead
  - Celery workers can call service functions directly

CONCEPT: asyncio.to_thread for blocking I/O
HuggingFace's load_dataset() is synchronous. Running it directly in an
async route would block FastAPI's event loop, freezing all other requests
until the download completes. asyncio.to_thread() offloads it to a thread
pool executor, keeping the event loop free.

File storage: uploads/datasets/ for now. In production, swap file_path
for an object storage key (S3/GCS) and update read/write accordingly.
"""
import asyncio, uuid
from pathlib import Path
from fastapi import UploadFile
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.exceptions import DatasetError, NotFoundError
from app.core.logging import get_logger
from app.models.dataset import DatasetORM
from app.schemas.dataset import DatasetFileCreate, DatasetHFCreate
from evalforge.datasets.loaders.csv_loader import load_csv, load_jsonl
from evalforge.datasets.loaders.huggingface_loader import load_from_huggingface

logger = get_logger(__name__)

UPLOAD_DIR = Path("uploads/datasets")
SUPPORTED_EXTENSIONS = {".csv", ".jsonl"}

def ensure_upload_dir() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

async def create_from_file(db: AsyncSession, file: UploadFile, metadata: DatasetFileCreate) -> DatasetORM:
    """
        Saves an uploaded CSV/JSONL, parses it, and registers the dataset in Postgres.

        Even if parsing fails, we persist the record with status="error" so the
        user can see what went wrong via GET /datasets/{id}. The file is always
        saved to disk regardless of parse success.
    """
    ensure_upload_dir()

    content = await file.read()
    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise DatasetError(
            f"Unsupported file type '{ext}'. Upload a .csv or .jsonl file."
        )

    source_type = "csv" if ext == ".csv" else "jsonl"

    # Save to disk with UUID filename to prevent collisions
    file_id = uuid.uuid4()
    file_path = UPLOAD_DIR / f"{file_id}{ext}"
    file_path.write_bytes(content)

    # Parse to validate format and count samples
    try:
        if source_type == "csv":
            samples = load_csv(content, metadata.name, metadata.format_config)
        else:
            samples = load_jsonl(content, metadata.name, metadata.format_config)
        status = "ready"
        error_message = None
        num_samples = len(samples)
    except ValueError as e:
        status = "error"
        error_message = str(e)
        num_samples = None
        logger.warning("dataset.parse_failed", filename=filename, error=str(e))

    dataset = DatasetORM(
        name=metadata.name,
        description=metadata.description,
        source_type=source_type,
        task_type=metadata.task_type,
        status=status,
        num_samples=num_samples,
        file_path=str(file_path),
        format_config=metadata.format_config,
        error_message=error_message,
    )
    db.add(dataset)
    await db.flush()

    logger.info(
        "dataset.created",
        id=str(dataset.id),
        source="file",
        name=metadata.name,
        status=status,
        num_samples=num_samples,
    )
    return dataset

async def create_from_huggingface(db: AsyncSession, metadata: DatasetHFCreate) -> DatasetORM:
    """
    Registers a HuggingFace Hub dataset. Downloads in a thread pool to avoid blocking the event loop.

    First call downloads to ~/.cache/huggingface/ and may take 10-60s.
    Subsequent calls for the same dataset ID + split are instant (cached).
    """
    try:
        samples = await asyncio.to_thread(
            load_from_huggingface,
            hf_dataset_id=metadata.hf_dataset_id,
            dataset_name=metadata.name,
            hf_config=metadata.hf_config,
            hf_split=metadata.hf_split,
            format_config=metadata.format_config,
        )
        status = "ready"
        error_message = None
        num_samples = len(samples)
    except Exception as e:
        status = "error"
        error_message = str(e)
        num_samples = None
        logger.warning(
            "dataset.hf_load_failed",
            hf_dataset_id=metadata.hf_dataset_id,
            error=str(e),
        )

    dataset = DatasetORM(
        name=metadata.name,
        description=metadata.description,
        source_type="huggingface",
        task_type=metadata.task_type,
        status=status,
        num_samples=num_samples,
        hf_dataset_id=metadata.hf_dataset_id,
        hf_config=metadata.hf_config,
        hf_split=metadata.hf_split,
        format_config=metadata.format_config,
        error_message=error_message,
    )
    db.add(dataset)
    await db.flush()

    logger.info(
        "dataset.created",
        id=str(dataset.id),
        source="huggingface",
        hf_dataset_id=metadata.hf_dataset_id,
        status=status,
        num_samples=num_samples,
    )
    return dataset

async def list_datasets(
    db: AsyncSession,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[DatasetORM], int]:
    """Return paginated list and total count, sorted newest first."""
    offset = (page - 1) * page_size

    rows = await db.execute(
        select(DatasetORM)
        .order_by(DatasetORM.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    items = list(rows.scalars().all())

    count = await db.execute(select(func.count()).select_from(DatasetORM))
    total = count.scalar_one()

    return items, total

async def get_dataset(db: AsyncSession, dataset_id: uuid.UUID) -> DatasetORM:
    """Fetch dataset by ID. Raises NotFoundError if missing."""
    result = await db.execute(
        select(DatasetORM).where(DatasetORM.id == dataset_id)
    )
    dataset = result.scalar_one_or_none()
    if dataset is None:
        raise NotFoundError(f"Dataset {dataset_id} not found")
    return dataset

async def delete_dataset(db: AsyncSession, dataset_id: uuid.UUID) -> None:
    """ Deletes dataset record and its uploaded file from disk. """
    dataset = await get_dataset(db, dataset_id)

    if dataset.file_path:
        p = Path(dataset.file_path)
        if p.exists():
            p.unlink()

    await db.delete(dataset)
    logger.info("dataset.deleted", id=str(dataset_id), name=dataset.name)

async def get_preview(db: AsyncSession, dataset_id: uuid.UUID, n: int = 5) -> tuple[DatasetORM, list]:
    """
    Returns the first N EvalSamples from a dataset for preview.

    Re-reads from disk or HF Hub — samples are never cached in memory between
    requests. For HF datasets, this uses the local cache (no re-download).
    """
    dataset = await get_dataset(db, dataset_id)

    if dataset.status != "ready":
        raise DatasetError(
            f"Dataset '{dataset.name}' is not ready (status: {dataset.status}). "
            + (f"Error: {dataset.error_message}" if dataset.error_message else "")
        )

    if dataset.source_type in ("csv", "jsonl"):
        file_path = Path(dataset.file_path)
        if not file_path.exists():
            raise DatasetError(
                f"Dataset file not found on disk: {dataset.file_path}. "
                f"The file may have been moved or deleted."
            )
        content = file_path.read_bytes()
        if dataset.source_type == "csv":
            samples = load_csv(content, dataset.name, dataset.format_config)
        else:
            samples = load_jsonl(content, dataset.name, dataset.format_config)

    else:  # huggingface
        samples = await asyncio.to_thread(
            load_from_huggingface,
            hf_dataset_id=dataset.hf_dataset_id,
            dataset_name=dataset.name,
            hf_config=dataset.hf_config,
            hf_split=dataset.hf_split or "test",
            n_samples=n,
            format_config=dataset.format_config,
        )

    return dataset, samples[:n]
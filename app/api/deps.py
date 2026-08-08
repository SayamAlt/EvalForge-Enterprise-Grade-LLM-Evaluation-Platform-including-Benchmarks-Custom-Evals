"""
FastAPI dependency injection.

CONCEPT: Dependency Injection
───────────────────────────────
FastAPI's `Depends()` system is DI — it resolves what a route needs
automatically. `Annotated[AsyncSession, Depends(get_db)]` tells FastAPI:
"call get_db(), await the generator, inject the result, clean up after."

This keeps route handlers clean:
    @router.get("/datasets")
    async def list_datasets(db: DbSession) -> list[DatasetRead]:
        ...  # db is already a live, auto-committing session

vs the alternative (boilerplate in every route):
    async with get_db_session() as db:
        ...
"""
from collections.abc import AsyncGenerator
from typing import Annotated
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db_session

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield a DB session for a single request. Auto-commits on success."""
    async with get_db_session() as session:
        yield session

# Type alias — use `DbSession` as the type hint in route signatures
DbSession = Annotated[AsyncSession, Depends(get_db)]
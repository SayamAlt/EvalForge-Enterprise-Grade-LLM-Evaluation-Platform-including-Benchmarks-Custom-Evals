"""
Async SQLAlchemy engine and session management.

CONCEPT: Async Database Sessions
──────────────────────────────────
Traditional ORMs use synchronous I/O — the thread blocks while waiting
for the DB. In a FastAPI app handling 100 concurrent eval requests, that
creates a thread-per-request bottleneck.

SQLAlchemy 2.0 with asyncpg solves this: every DB call is a coroutine,
so the event loop handles other requests while waiting for Postgres.

Session lifecycle:
  1. Request comes in → FastAPI calls get_db() dependency
  2. A new AsyncSession is pulled from the connection pool
  3. Route handler runs, performs queries
  4. On success: session.commit() flushes changes
  5. On exception: session.rollback() undoes partial changes
  6. Session returns to pool (not closed — pooling reuses connections)

Never import `engine` directly in routes — always use the `get_db` dependency
in app/api/deps.py which handles the full lifecycle.
"""
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine
)
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Module-level singletons — initialized once at startup via init_db()
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None

def _create_engine() -> AsyncEngine:
    return create_async_engine(
        settings.database_url,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        pool_pre_ping=True, # Verify connection health before use
        echo=settings.app_debug
    )

def _create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,  # Do not reload objects after commit
        autocommit=False,
        autoflush=False
    )

async def init_db() -> None:
    """ Initializes the database engine and session factory. Called at startup. """
    global _engine, _session_factory
    _engine = _create_engine()
    _session_factory = _create_session_factory(_engine)
    # Log without exposing credentials
    safe_url = settings.database_url.split("@")[-1] if "@" in settings.database_url else "***"
    logger.info("database.initialized", url=safe_url)

async def close_db() -> None:
    """ Disposes the engine and releases all connections. Called at shutdown. """
    global _engine
    
    if _engine:
        await _engine.dispose()
        logger.info("database.closed")

@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager that yields a database session with auto commit/rollback.

    Usage (outside of FastAPI dependency injection):
        async with get_db_session() as session:
            result = await session.execute(select(Dataset))
    """
    if _session_factory is None:
        raise RuntimeError("Database not initialized. Ensure init_db() ran at startup.")
    
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
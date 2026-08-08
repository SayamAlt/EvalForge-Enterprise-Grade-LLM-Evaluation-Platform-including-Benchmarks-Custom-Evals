"""
Pytest fixtures shared across all tests.

CONCEPT: Test Fixtures
────────────────────────
pytest fixtures are setup/teardown functions injected into tests by name.
Session-scoped fixtures (scope="session") run once for the entire test suite.
Function-scoped fixtures (default) run fresh for each test.

This conftest provides:
  - db_engine   → creates test tables on session start, drops on session end
  - db_session  → one clean session per test, rolls back after each test
  - client      → httpx AsyncClient pointing at the FastAPI test app

Pattern for isolation: each test gets a fresh session that rolls back,
so tests don't pollute each other's data even without explicit cleanup.
"""
import asyncio, pytest
from collections.abc import AsyncGenerator
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from app.core.config import settings
from app.models.base import Base

# Use a separate test database to avoid clobbering dev data
TEST_DATABASE_URL = settings.database_url.replace("/evalforge", "/evalforge_test")

@pytest.fixture(scope="session")
def event_loop():
    """ Creates a single event loop for the entire test session. """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def db_engine():
    """ Creates all tables before tests, drop them after. """
    engine = create_async_engine(TEST_DATABASE_URL, poolclass=NullPool)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
    yield engine
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        
    await engine.dispose()

@pytest.fixture
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    """
        Yields a clean DB session for one test.
        Rolls back after the test so no data leaks between tests.
    """
    factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    
    async with factory() as session:
        yield session
        await session.rollback()

@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """ AsyncClient hitting the FastAPI app directly (no network). """
    from app.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as c:
        yield c
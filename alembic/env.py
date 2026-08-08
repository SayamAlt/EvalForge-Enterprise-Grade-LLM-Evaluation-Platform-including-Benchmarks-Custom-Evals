"""
Alembic migration environment.

CONCEPT: Database Migrations

Every schema change (add column, new table, drop index) is a migration:
a versioned Python file that knows how to upgrade and downgrade the DB.

Why migrations instead of `CREATE TABLE IF NOT EXISTS`?
  - Version-controlled schema history
  - Reproducible across dev / staging / prod
  - Rollback on bad deployments
  - Team members get schema changes via `alembic upgrade head`

Workflow:
    # Auto-generate migration from model changes
    alembic revision --autogenerate -m "add dataset table"

    # Apply all pending migrations
    alembic upgrade head

    # Roll back last migration
    alembic downgrade -1

This file configures alembic to use:
  - Async engine (asyncpg) for online migrations
  - Sync URL for offline mode (used in CI without a live DB)
  - app.models Base.metadata so autogenerate sees all ORM models
"""
import asyncio
from logging.config import fileConfig
from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine
from app.core.config import settings
# Import Base so all models are registered in Base.metadata
from app.models.base import Base  # noqa: F401
import app.models  # noqa: F401 which triggers all model imports

config = context.config
config.set_main_option("sqlalchemy.url", settings.sync_database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """ Emits SQL to stdout without a live DB connection (used in CI). """
    url = config.get_main_option("sqlalchemy.url")
    
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"}
    )
    
    with context.begin_transaction():
        context.run_migrations()

def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    
    with context.begin_transaction():
        context.run_migrations()

async def run_async_migrations() -> None:
    """ Runs migrations against a live async DB connection. """
    connectable = create_async_engine(
        settings.database_url,  # uses +asyncpg; sync URL in config is for offline mode only
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()

def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
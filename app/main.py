"""
FastAPI application factory.

CONCEPT: Application Factory Pattern
──────────────────────────────────────
We don't instantiate `app = FastAPI()` at module level with all config inline.
Instead, `create_app()` builds and returns the configured app. Benefits:
  - Testable: tests can call create_app() with different settings
  - Clear startup/shutdown: lifespan context manager owns both
  - Middleware, routers, handlers all wired in one place

Startup sequence (via lifespan):
  1. setup_logging() — configure structlog
  2. init_db()       — create SQLAlchemy engine + session factory
  [app serves requests]
  3. close_db()      — dispose connection pool gracefully
"""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import v1_router
from app.core.config import settings
from app.core.database import close_db, init_db
from app.core.exceptions import EvalForgeError
from app.core.logging import get_logger, setup_logging

# Logging must be configured before any other module uses get_logger()
setup_logging(log_level=settings.log_level, log_format=settings.log_format)
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Async context manager that owns app startup and shutdown."""
    logger.info("evalforge.starting", env=settings.app_env, debug=settings.app_debug)
    await init_db()
    yield
    logger.info("evalforge.shutting_down")
    await close_db()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description=(
            "Enterprise-grade LLM Evaluation & Benchmarking Platform. "
            "Evaluate models, track experiments, and measure quality at scale."
        ),
        version="0.1.0",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # ── Middleware ──────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception handlers ──────────────────────────────────────────────────
    @app.exception_handler(EvalForgeError)
    async def evalforge_error_handler(
        request: Request, exc: EvalForgeError
    ) -> JSONResponse:
        logger.warning(
            "http.error",
            status=exc.status_code,
            detail=exc.detail,
            path=str(request.url),
            method=request.method,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail, "type": exc.__class__.__name__},
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.error(
            "http.unhandled_error",
            error=str(exc),
            path=str(request.url),
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error.", "type": "InternalError"},
        )

    # ── Routers ─────────────────────────────────────────────────────────────
    app.include_router(v1_router, prefix="/api/v1")

    return app


app = create_app()

"""
Structured logging via structlog.

CONCEPT: Structured Logging
────────────────────────────
Traditional logging:  logger.info("User logged in with id 42")
Structured logging:   logger.info("user.login", user_id=42)

Why structured?
  - Machine-parseable JSON → works with Datadog, CloudWatch, Loki, Splunk
  - Every log entry is a dict → filterable, aggregatable, queryable
  - Context variables (request_id, run_id) auto-attach to all logs in scope

In production (LOG_FORMAT=json):
  {"event": "user.login", "user_id": 42, "level": "info", "timestamp": "..."}

In development (LOG_FORMAT=console):
  2024-01-01 12:00:00 [info] user.login [user_id=42]

Usage:
    from app.core.logging import get_logger
    logger = get_logger(__name__)
    logger.info("evaluator.run.start", run_id=run_id, model=model)
    logger.error("provider.call.failed", error=str(exc), model=model)
"""
import logging, sys, structlog
from typing import Any
from structlog.types import EventDict, WrappedLogger

def _add_app_context(logger: WrappedLogger, method: str, event_dict: EventDict) -> EventDict:
    event_dict["app"] = "evalforge"
    return event_dict

def setup_logging(log_level: str = "INFO", log_format: str = "json") -> None:
    """
    Configure structlog + stdlib logging. Call once at application startup.

    Args:
        log_level:  "DEBUG" | "INFO" | "WARNING" | "ERROR"
        log_format: "json" (production) | "console" (development)
    """
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        _add_app_context
    ]

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if log_format == "json"
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "anthropic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """ Returns a bound structlog logger for the given module name. """
    return structlog.get_logger(name)
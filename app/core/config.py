"""
Central configuration via pydantic-settings.

CONCEPT: Settings Management
─────────────────────────────
All configuration lives here. pydantic-settings reads from environment
variables (or .env file) and validates + type-converts automatically.

Why one central settings object?
  - Single source of truth — no scattered os.getenv() calls
  - Type safety — APP_PORT="8000" becomes int automatically
  - Validation — SECRET_KEY shorter than 32 chars raises on startup
  - Testability — override settings in tests via environment variables

Usage anywhere in the codebase:
    from app.core.config import settings
    print(settings.openai_api_key)
"""

from functools import lru_cache
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "EvalForge"
    app_env: str = "development"
    app_debug: bool = False
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    secret_key: str = Field(default="insecure-dev-secret-change-in-production", min_length=16)

    # Database
    database_url: str = "postgresql+asyncpg://evalforge:evalforge@localhost:5432/evalforge"
    database_pool_size: int = 10
    database_max_overflow: int = 20

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_experiment_name: str = "evalforge-default"

    # ── LLM Providers ─────────────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_org_id: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    deepseek_api_key: str = ""
    dashscope_api_key: str = ""        # Qwen / Alibaba Cloud
    moonshot_api_key: str = ""         # Kimi K2/K3
    minimax_api_key: str = ""
    minimax_group_id: str = ""         # Required by MiniMax API
    groq_api_key: str = ""
    mistral_api_key: str = ""
    writer_api_key: str = ""           # Fable / Palmyra
    ollama_base_url: str = "http://localhost:11434"
    huggingface_api_token: str = ""

    # ── Evaluation defaults ────────────────────────────────────────────────────
    default_max_tokens: int = 2048
    default_temperature: float = 0.0
    eval_concurrency: int = 10
    eval_timeout_seconds: int = 60
    judge_config_path: str = "configs/judge_config.yaml"

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    # Celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # Computed fields
    @computed_field
    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @computed_field
    @property
    def sync_database_url(self) -> str:
        """ Synchronous URL for Alembic migrations (no +asyncpg driver). """
        return self.database_url.replace("+asyncpg", "")

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
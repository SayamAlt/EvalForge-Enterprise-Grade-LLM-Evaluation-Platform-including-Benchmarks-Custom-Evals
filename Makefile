.PHONY: help install install-all dev-up dev-down migrate migrate-create test test-unit test-integration test-providers test-providers-fast lint format check clean

# Variables
UV := uv
PYTHON := python
APP_DIR := app
CORE_DIR := evalforge
TEST_DIR := tests

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

install: ## Install core dependencies (dev + prod)
	$(UV) pip install -e ".[dev]"

install-all: ## Install all optional extras (metrics, HF, tracking, etc.)
	$(UV) pip install -e ".[dev,ollama,hf,metrics,semantic,langchain,tracking,ops]"

dev-up: ## Start all Docker services
	docker compose up -d
	@echo "Services up:"
	@echo "  API:    http://localhost:8000/docs"
	@echo "  MLflow: http://localhost:5000"

dev-down: ## Stop all Docker services
	docker compose down

dev-logs: ## Tail logs for all services
	docker compose logs -f

migrate: ## Run pending DB migrations
	alembic upgrade head

migrate-create: ## Create a new migration (usage: make migrate-create MSG="add dataset table")
	alembic revision --autogenerate -m "$(MSG)"

migrate-down: ## Roll back the last migration
	alembic downgrade -1

test: ## Run full test suite with coverage
	pytest $(TEST_DIR)/ -v --cov=$(APP_DIR) --cov=$(CORE_DIR) --cov-report=term-missing

test-unit: ## Run unit tests only
	pytest $(TEST_DIR)/unit/ -v

test-integration: ## Run integration tests only
	pytest $(TEST_DIR)/integration/ -v

test-providers: ## Live smoke test — calls real provider APIs (needs .env keys)
	$(UV) run python scripts/smoke_test_providers.py

test-providers-fast: ## Live smoke test for cheap/fast providers only
	$(UV) run python scripts/smoke_test_providers.py --providers openai anthropic groq deepseek

lint: ## Run linter (ruff)
	ruff check $(APP_DIR)/ $(CORE_DIR)/ $(TEST_DIR)/

format: ## Format code (ruff)
	ruff format $(APP_DIR)/ $(CORE_DIR)/ $(TEST_DIR)/
	ruff check --fix $(APP_DIR)/ $(CORE_DIR)/ $(TEST_DIR)/

check: lint ## Run all static analysis checks
	mypy $(APP_DIR)/ $(CORE_DIR)/

clean: ## Remove cache and build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -name "*.pyc" -delete
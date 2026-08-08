# EvalForge

Enterprise-grade LLM Evaluation & Benchmarking Platform.

Systematically measure, compare, and optimize the performance of LLMs, RAG
systems, and AI agents across standard benchmarks and custom datasets.

## What It Does

- **Model Evals** — Benchmark models on MMLU, GSM8K, HellaSwag, HumanEval, TruthfulQA
- **Custom Evals** — Evaluate on your own CSV/JSONL datasets with any metric
- **Multi-provider** — OpenAI, Anthropic, Gemini, Ollama, HuggingFace from one interface
- **Metrics** — Exact Match, ROUGE, BLEU, BERTScore, LLM-as-a-Judge, Latency, Cost
- **Experiment Tracking** — MLflow integration with leaderboards and comparison views
- **Application Evals** — RAG, SQL generation, agent evaluation (later sprints)

## Tech Stack

| Layer              | Technology                              |
|--------------------|------------------------------------------|
| API                | FastAPI + Uvicorn                        |
| Task Queue         | Celery + Redis                           |
| Database           | PostgreSQL + SQLAlchemy (async)          |
| Migrations         | Alembic                                  |
| Experiment Tracking| MLflow                                   |
| LLM Providers      | OpenAI, Anthropic, Google, Ollama        |
| Eval Framework     | HuggingFace Evaluate + custom            |
| Deployment         | Docker + Docker Compose                  |

## Project Structure

```
evalforge/
├── app/                    # FastAPI service
│   ├── main.py             # App factory + lifespan
│   ├── core/               # Config, logging, DB, exceptions
│   ├── api/v1/             # REST endpoints
│   ├── models/             # SQLAlchemy ORM models
│   ├── schemas/            # Pydantic request/response schemas
│   ├── services/           # Business logic
│   └── workers/            # Celery background tasks
│
├── evalforge/              # Core evaluation library
│   ├── providers/          # LLM provider abstractions (OpenAI, Anthropic, ...)
│   ├── datasets/           # Dataset loaders + benchmark definitions
│   ├── metrics/            # Metric implementations (BLEU, ROUGE, BERTScore, ...)
│   ├── evaluators/         # Evaluation orchestration
│   └── tracking/           # MLflow / experiment tracking
│
├── alembic/                # Database migrations
├── docker/                 # Dockerfiles
├── tests/                  # Unit + integration tests
├── notebooks/              # Educational Jupyter notebooks
├── scripts/                # CLI evaluation scripts
└── configs/                # YAML configuration files
```

## Quick Start

```bash
# 1. Clone and install
git clone <repo>
cd evalforge
cp .env.example .env          # add your API keys
make install

# 2. Start infrastructure
make dev-up                   # starts Postgres, Redis, MLflow

# 3. Run migrations
make migrate

# 4. Start the API
uvicorn app.main:app --reload

# 5. Open docs
open http://localhost:8000/docs
open http://localhost:5000      # MLflow UI
```

## Sprint Roadmap

| Sprint | Feature                          | Status      |
|--------|----------------------------------|-------------|
| 1      | Project setup, FastAPI, DB       | ✅ Complete  |
| 2      | Dataset Manager                  | ✅ Complete  |
| 3      | Model Provider abstraction       | 🔜 Next      |
| 4      | Evaluation Engine                | 📋 Planned   |
| 5      | Experiment Tracking              | 📋 Planned   |
| 6      | LLM-as-a-Judge                   | 📋 Planned   |
| 7      | RAG Evaluation                   | 📋 Planned   |
| 8      | SQL Evaluation                   | 📋 Planned   |
| 9      | Agent Evaluation (LangGraph)     | 📋 Planned   |
| 10     | Dashboard, CI/CD, Deployment     | 📋 Planned   |

## Running Tests

```bash
make test            # full suite with coverage
make test-unit       # unit tests only (no DB required)
make test-integration
```
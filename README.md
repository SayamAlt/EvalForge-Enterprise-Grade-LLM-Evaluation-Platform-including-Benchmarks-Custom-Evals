# EvalForge

**Enterprise-grade LLM Evaluation & Benchmarking Platform**

Systematically evaluate, compare, and rank LLMs across standard benchmarks and custom datasets — with a single REST API, built-in experiment tracking, and support for 39 models across 12 providers out of the box.

---

## Features

| Feature | Details |
|---|---|
| **Multi-provider** | 39 models across OpenAI, Anthropic, Google, Groq, Mistral, DeepSeek, Qwen, Moonshot (Kimi), MiniMax, Writer, Ollama |
| **Benchmarks** | GSM8K, MMLU (full), HellaSwag, TruthfulQA, HLE, HumanEval-style code evals |
| **Custom Datasets** | Upload CSV or JSONL — any column mapping, any metric |
| **Metrics** | `answer_match`, `choice_match`, `multi_choice_match`, `numeric_match`, `rouge_l`, `code_match`, `label_match` |
| **Batch Evaluation** | Evaluate N models in parallel with concurrency control, retry logic, and fast-abort on fatal errors |
| **Experiment Tracking** | Create experiments, attach runs, view leaderboards, compare runs side-by-side |
| **Cost & Latency** | Per-sample cost and latency tracked for every run |
| **Leaderboard Export** | CSV leaderboard with best-per-model deduplication |
| **Background Execution** | Long evals run async — API returns 202 immediately |
| **Health Checks** | Ping individual models or all registered models at once |

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Database | PostgreSQL + SQLAlchemy (async) |
| Migrations | Alembic |
| Cache / Broker | Redis |
| LLM Providers | OpenAI, Anthropic, Google, Groq, Mistral, DeepSeek, Qwen, Moonshot, MiniMax, Writer, Ollama |
| Dataset Loading | HuggingFace Datasets + CSV/JSONL loaders |
| Deployment | Docker + Docker Compose |

---

## API Endpoints

### Models
| Method | Route | Description |
|---|---|---|
| `GET` | `/api/v1/models` | List all registered models |
| `POST` | `/api/v1/models` | Register a new model |
| `POST` | `/api/v1/models/ping` | Ping a single model |
| `POST` | `/api/v1/models/ping-all` | Ping all models and report availability |

### Datasets
| Method | Route | Description |
|---|---|---|
| `GET` | `/api/v1/datasets` | List all datasets |
| `POST` | `/api/v1/datasets` | Upload CSV/JSONL dataset |
| `GET` | `/api/v1/datasets/{id}` | Dataset details |
| `DELETE` | `/api/v1/datasets/{id}` | Delete dataset |

### Benchmarks
| Method | Route | Description |
|---|---|---|
| `GET` | `/api/v1/benchmarks` | List registered benchmarks |
| `POST` | `/api/v1/benchmarks` | Register HuggingFace benchmark dataset |

### Evaluations
| Method | Route | Description |
|---|---|---|
| `POST` | `/api/v1/evaluations` | Run single-model evaluation |
| `POST` | `/api/v1/evaluations/batch` | Run multi-model evaluation in parallel |
| `GET` | `/api/v1/evaluations/{id}` | Get evaluation run status and results |

### Experiments
| Method | Route | Description |
|---|---|---|
| `GET` | `/api/v1/experiments` | List all experiments |
| `POST` | `/api/v1/experiments` | Create experiment |
| `GET` | `/api/v1/experiments/{id}` | Experiment details |
| `GET` | `/api/v1/experiments/{id}/runs` | All runs attached to experiment |
| `POST` | `/api/v1/experiments/{id}/runs` | Attach a run to experiment |
| `GET` | `/api/v1/experiments/{id}/leaderboard` | Model leaderboard for experiment |
| `POST` | `/api/v1/experiments/compare` | Compare multiple runs side-by-side |

### Exports
| Method | Route | Description |
|---|---|---|
| `GET` | `/api/v1/exports/leaderboard` | Export global leaderboard as CSV |

---

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/SayamAlt/EvalForge-Enterprise-Grade-LLM-Evaluation-Platform-including-Benchmarks-Custom-Evals
cd evalforge
cp .env.example .env     # add your API keys

# 2. Start all services
docker compose up -d --build

# 3. Open API docs
open http://localhost:8000/docs
```

---

## Project Structure

```
evalforge/
├── app/                        # FastAPI service
│   ├── main.py                 # App factory + lifespan
│   ├── core/                   # Config, logging, DB, exceptions
│   ├── api/v1/endpoints/       # REST endpoints
│   │   ├── models.py
│   │   ├── datasets.py
│   │   ├── benchmarks.py
│   │   ├── evaluations.py
│   │   ├── experiments.py
│   │   ├── exports.py
│   │   └── health.py
│   ├── models/                 # SQLAlchemy ORM models
│   ├── schemas/                # Pydantic request/response schemas
│   └── services/               # Business logic
│
├── evalforge/                  # Core evaluation library
│   ├── providers/              # LLM provider clients (11 providers)
│   ├── datasets/               # Dataset loaders + benchmark definitions
│   │   ├── loaders/            # CSV, JSONL, HuggingFace loaders
│   │   └── benchmarks/         # Benchmark dataset configs
│   └── metrics/                # Metric implementations
│       ├── text/               # answer_match, choice_match, rouge_l, code_match, numeric_match, ...
│       ├── semantic/           # Semantic similarity metrics
│       ├── llm_judge/          # LLM-as-a-judge scoring
│       └── production/         # Production eval metrics
│
├── configs/
│   ├── model_config.yaml       # 39 model definitions + pricing
│   ├── eval_config.yaml        # Evaluation defaults
│   └── judge_config.yaml       # LLM-judge configuration
│
├── alembic/                    # Database migrations
├── docker/                     # Dockerfiles
└── results/                    # Generated leaderboard CSVs
```

---

## Supported Models (39)

| Provider | Models |
|---|---|
| **OpenAI** | gpt-4o, gpt-4o-mini, gpt-4-turbo, o3-mini |
| **Anthropic** | claude-opus-4-7, claude-sonnet-4-6, claude-haiku-4-5 |
| **Google** | gemini-2.0-flash, gemini-2.5-flash, gemini-2.5-pro |
| **Groq** | llama-3.3-70b, llama-3.1-8b, llama-4-scout, qwen3-27b, gpt-oss-120b, gpt-oss-20b |
| **Mistral** | mistral-large-2, mistral-large-latest, mistral-medium-3, mistral-small, ministral-8b, magistral-small |
| **DeepSeek** | deepseek-v3, deepseek-r1 |
| **Qwen** | qwen3-235b, qwen3-30b |
| **Moonshot** | kimi-k2, kimi-k3 |
| **MiniMax** | minimax-m1, minimax-m3 |
| **Writer** | palmyra-x-004 |
| **Ollama** | deepseek-r1:7b, qwen2.5:72b, qwen2.5:7b, mistral-nemo:12b, phi4:14b, llama3:8b, llama3:70b, mistral:7b |

---

## Evaluation Metrics

| Metric | Task Type | Description |
|---|---|---|
| `answer_match` | Open-ended QA | Exact → contains (with verbosity penalty) → token F1 |
| `choice_match` | Single-choice MCQ | Letter extraction (A/B/C/D) → choice text → reference fallback |
| `multi_choice_match` | Multi-choice MCQ | Jaccard similarity between predicted and expected letter sets |
| `numeric_match` | Math QA | Extracts last number from CoT output, compares as float |
| `rouge_l` | Summarization / Generation | Rouge-L F1 via longest common subsequence |
| `code_match` | Code generation | Token F1 after stripping comments and normalizing whitespace |
| `label_match` | Classification | Exact → whole-word boundary match |

---

## Experiment Workflow

```
1. POST /experiments              → create experiment (container for related runs)
2. POST /evaluations/batch        → run models on dataset → returns run_ids
3. POST /experiments/{id}/runs    → attach each run to the experiment
4. GET  /experiments/{id}/runs    → list all runs with status + metrics
5. GET  /experiments/{id}/leaderboard → ranked model comparison
6. POST /experiments/compare      → detailed side-by-side sample-level comparison
```

---

## Environment Variables

```env
# Database
DATABASE_URL=postgresql+asyncpg://evalforge:evalforge@postgres:5432/evalforge

# Redis
REDIS_URL=redis://redis:6379/0

# LLM Provider API Keys
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
GROQ_API_KEY=
MISTRAL_API_KEY=
DEEPSEEK_API_KEY=
DASHSCOPE_API_KEY=        # Qwen
MOONSHOT_API_KEY=         # Kimi
MINIMAX_API_KEY=
WRITER_API_KEY=
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

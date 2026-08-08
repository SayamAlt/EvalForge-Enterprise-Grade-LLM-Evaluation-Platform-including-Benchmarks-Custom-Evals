FROM python:3.11-slim

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy dependency spec first (Docker layer caching — deps reinstall only if pyproject.toml changes)
COPY pyproject.toml README.md ./

# Install all dependencies
RUN uv pip install --system --no-cache -e ".[dev,hf]"

# Copy application source
COPY . .

# Non-root user (security best practice)
RUN useradd -m -u 1000 evalforge && chown -R evalforge:evalforge /workspace
USER evalforge

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
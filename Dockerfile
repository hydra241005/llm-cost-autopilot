### Multi-stage Dockerfile for API and worker
### Usage:
###  docker build -t llm-cost-autopilot:api --target=api .
###  docker build -t llm-cost-autopilot:worker --target=worker .

FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install runtime dependencies from pyproject
COPY pyproject.toml pyproject.toml
COPY src src

RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir .

FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "autopilot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM base AS worker
CMD ["python", "-m", "autopilot.scripts.worker"]

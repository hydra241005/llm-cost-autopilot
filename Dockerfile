### Production-ready multi-stage Dockerfile
### Targets: builder -> runtime

FROM python:3.11-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl ca-certificates gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy dependency declarations first for layer caching
COPY pyproject.toml pyproject.toml
COPY src src

RUN python -m pip install --upgrade pip setuptools wheel poetry && \
    python -m pip wheel . -w /wheels

FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create non-root user
RUN groupadd --gid 1000 autopilot && useradd --uid 1000 --gid autopilot --shell /bin/sh -m autopilot

WORKDIR /app

# Copy wheels from builder and install
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy only application code
COPY src src

# Expose port and set non-root user
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 CMD curl -f http://localhost:8000/health || exit 1

USER autopilot

ENV PATH="/home/autopilot/.local/bin:${PATH}"

CMD ["uvicorn", "autopilot.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "30"]

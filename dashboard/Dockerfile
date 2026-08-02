FROM python:3.11-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml pyproject.toml
COPY src src
COPY dashboard dashboard

# Copy entrypoint and make executable
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

RUN python -m pip install --upgrade pip setuptools wheel && python -m pip wheel . -w /wheels

FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN groupadd --gid 1000 autopilot && useradd --uid 1000 --gid autopilot --shell /bin/sh -m autopilot

WORKDIR /app
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-cache-dir /wheels/* && rm -rf /wheels

COPY src src
COPY dashboard dashboard

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 CMD curl -f http://localhost:8501/ || exit 1

USER autopilot
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["streamlit", "run", "dashboard/app.py", "--server.port", "8501", "--server.headless", "true"]

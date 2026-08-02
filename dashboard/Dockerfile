FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml pyproject.toml
COPY src src
COPY dashboard dashboard

RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir .[dashboard]

EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.port", "8501", "--server.headless", "true"]

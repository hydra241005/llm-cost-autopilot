.PHONY: help install dev test lint typecheck check baseline doctor clean

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install:  ## Create the virtualenv and install all extras
	uv venv --python 3.11
	uv pip install -e ".[dev,ml,db]"

dev:  ## Run the API with autoreload
	uv run uvicorn autopilot.api.main:get_app --factory --reload

test:  ## Run the test suite
	uv run pytest

lint:  ## Check formatting and lint rules
	uv run ruff check .

typecheck:  ## Run static type checking
	uv run mypy

check: lint typecheck test  ## Run the full gate

baseline:  ## Fan the sample prompts across every model and write data/baseline_run.json
	uv run python scripts/baseline_run.py

doctor:  ## Check providers, Ollama, and pulled models
	uv run python scripts/doctor.py

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage build dist

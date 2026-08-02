# LLM Cost Autopilot

An intelligent routing layer for LLM traffic. It classifies each request's complexity, routes it to the cheapest model that can actually handle it, verifies quality asynchronously on a sampled basis, and retrains its own router from verified failures.

> **Status:** Phase 1 complete — provider abstraction, three adapters, unified request interface, model registry. See [`docs/`](docs/) for the full engineering blueprint (10 documents) and [`docs/10-roadmap-testing-final-plan.md`](docs/10-roadmap-testing-final-plan.md) for the milestone plan.

## Why it exists

Most production LLM traffic is simple: extraction, formatting, short answers. Sending all of it to a premium model is the single most common source of avoidable spend. Routing by complexity fixes that — but only if you can prove quality did not degrade, which is why verification and a savings figure *net of verification cost* are first-class parts of the design rather than an afterthought.

## Architecture

Clean architecture, ports and adapters. The dependency rule is enforced by convention and an import contract in CI:

```
api ──▶ application ──▶ domain ◀── infrastructure ◀── worker
```

- **`domain/`** — entities, enums, errors, ports, and pure policies. Imports only the standard library and pydantic. No SQLAlchemy, no FastAPI, no vendor SDK.
- **`application/`** — use cases that orchestrate domain logic over injected ports.
- **`infrastructure/`** — adapters implementing those ports: provider SDKs, persistence, cache, resilience, observability.
- **`api/`, `worker/`** — the only composition roots; the only places concrete adapters are constructed.

Every port has a real implementation and a test fake, which is why the test suite runs with no network and no database.

## Quickstart

```bash
uv venv --python 3.11 && uv pip install -e ".[dev,ml,db]"
```

```bash
cp .env.example .env
```

Nothing in `.env` is required. With no credentials at all the stack still runs against a local Ollama model — that is the free Tier 1 floor, and it exists so a reviewer can clone the repo and see it work.

```bash
uv run uvicorn autopilot.api.main:get_app --factory --reload
```

Then visit `http://localhost:8000/docs` for interactive OpenAPI documentation.

### Verifying your environment

```bash
uv run python scripts/doctor.py
```

The doctor reports which providers have credentials, whether Ollama is running, and which local models still need pulling — naming the exact `ollama pull` command for each. Add `--pull` to download them (several GB each). Setting `OLLAMA_AUTO_PULL=true` does the same automatically at API startup; it is off by default because consuming gigabytes of disk during boot should be an explicit choice.

A missing local model is also caught at request time: the adapter maps Ollama's 404 to a message naming the pull command, and `GET /health` carries a `setup_warning` field while the local tier is unusable.

### Task runner (Windows-friendly)

`make` is not installed on most Windows machines, so every Makefile target has an identical cross-platform equivalent:

```bash
uv run python scripts/task.py check
```

| Target | Make | Cross-platform | What it does |
|---|---|---|---|
| install | `make install` | `uv run python scripts/task.py install` | Create the venv, install all extras |
| dev | `make dev` | `uv run python scripts/task.py dev` | Run the API with autoreload |
| test | `make test` | `uv run python scripts/task.py test` | Run the test suite |
| lint | `make lint` | `uv run python scripts/task.py lint` | Ruff |
| typecheck | `make typecheck` | `uv run python scripts/task.py typecheck` | mypy |
| check | `make check` | `uv run python scripts/task.py check` | The full gate: lint, typecheck, test |
| baseline | `make baseline` | `uv run python scripts/task.py baseline` | Fan sample prompts across every model |
| doctor | `make doctor` | `uv run python scripts/task.py doctor` | Environment preflight |
| clean | `make clean` | `uv run python scripts/task.py clean` | Remove caches and build artifacts |

## Operations

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness. Always 200 when the process can serve traffic; carries `setup_warning` when a provider is reachable but unusable |
| `GET /health/providers` | Per-provider circuit state plus rolling metrics: availability, success rate, timeout and 429 counts, p50/p95 latency |
| `GET /v1/models` | The model catalogue, filterable by tier |
| `GET /admin/metrics` | Aggregated routing, shadow, A/B, and classifier lifecycle metrics |
| `GET /admin/routing/explain/{request_id}` | Explainability payload for a recorded routing decision |
| `GET /admin/classifiers/compare` | Comparison overview for classifier versions |

`/health/providers` returns 200 even when providers are degraded — it describes *upstream* health, and a non-200 would make an orchestrator restart a perfectly healthy process because a vendor was down. Read the per-provider `state` and `healthy` fields instead.

A provider's circuit opens only when it is **both** meaningfully broken (≥ `ROUTING_BREAKER_FAILURE_THRESHOLD` failures) **and** broken at a meaningful rate (≥ `ROUTING_BREAKER_FAILURE_RATE` of recent calls) inside a rolling window. Requiring both prevents a burst of five failures among a thousand healthy calls from cutting off a working provider. Malformed requests never trip a circuit: they would fail identically against a healthy vendor. Idle providers report `null` rates rather than a misleading 100%.

## Configuration

All configuration is environment-driven through a single `AppSettings` object (`src/autopilot/config.py`); no module outside that file reads `os.environ`. Secrets are wrapped in `SecretStr` so they cannot reach a log line or a traceback.

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | unset | Enables the OpenAI adapter |
| `ANTHROPIC_API_KEY` | unset | Enables the Anthropic adapter |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local model server |
| `OLLAMA_AUTO_PULL` | `false` | Download missing local models at startup |
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/autopilot.db` | SQLite by default; set a `postgresql+asyncpg://` URL to switch dialects |
| `REDIS_URL` | unset | Optional. When unset, in-process cache, queue, and rate-limit adapters are used |
| `ROUTING_CONFIDENCE_THRESHOLD` | `0.6` | Below this confidence the effective tier is raised by one |
| `ROUTING_BREAKER_FAILURE_THRESHOLD` | `5` | Failures in the window before a circuit may open |
| `ROUTING_BREAKER_FAILURE_RATE` | `0.5` | Failure share required alongside the count |
| `ROUTING_METRICS_WINDOW_S` | `300` | Rolling window for `/health/providers` figures |
| `LOG_LEVEL` / `LOG_JSON` | `INFO` / `true` | Structured logging |

Model pricing and tiers live in [`configs/models.yaml`](configs/models.yaml); the tier-to-model map and fallback chains live in [`configs/routing.yaml`](configs/routing.yaml). Both are seed configuration — reviewable in git, imported into the database on first boot, versioned through the API thereafter.

## Testing

```bash
uv run pytest
```

```bash
uv run ruff check . && uv run mypy
```

Tests requiring real provider credentials are marked `live` and skipped by default:

```bash
uv run pytest -m live
```

## Design decisions

1. **Sampled verification, not exhaustive.** A verifier that runs on every request erases the savings it exists to measure. Sampling is budgeted, and the headline metric is savings *net of verification spend*.
2. **SQLite by default, PostgreSQL supported.** One SQLAlchemy 2.0 async codebase, dual-dialect. Postgres is one environment variable away, and migrations stay compatible with both.
3. **Redis optional, behind ports.** Queue, cache, rate limiting, and breaker state are domain ports with in-process implementations. The stack runs end to end with no Redis; Compose still provides it.
4. **Two escalation modes.** A synchronous guardrail may re-route once *before* the response is delivered. Asynchronous learning happens *after* delivery and only ever produces training data — a delivered response is never quietly swapped.
5. **Official vendor SDKs inside adapters.** The `anthropic` and `openai` SDKs absorb vendor API drift; Ollama, which has no SDK, uses `httpx`. The port boundary makes the difference invisible to callers.

## Documentation

| Doc | Contents |
|---|---|
| [01 — Architecture Review](docs/01-architecture-review.md) | Weaknesses, fixes, risk register |
| [02 — System Architecture](docs/02-system-architecture.md) | Diagrams, request lifecycle |
| [03 — Folder Structure & Modules](docs/03-folder-structure-modules.md) | Layout, ports, dependency rules |
| [04 — Database Schema](docs/04-database-schema.md) | Tables, indexes, migrations |
| [05 — API Specification](docs/05-api-specification.md) | Every endpoint and error |
| [06 — Routing Engine & Classifier](docs/06-routing-engine-classifier.md) | Registry, resilience, features, retraining |
| [07 — Verification & Observability](docs/07-verification-and-observability.md) | Judge design, logging, metrics |
| [08 — Dashboard UI/UX](docs/08-dashboard-uiux.md) | Design system, all pages |
| [09 — Docker & Deployment](docs/09-docker-and-deployment.md) | Compose topology, six deploy targets |
| [10 — Roadmap & Testing](docs/10-roadmap-testing-final-plan.md) | Milestones, testing strategy |
| [11 — Shadow Evaluation](docs/11-shadow-evaluation-architecture.md) | Candidate shadow-mode evaluation design |
| [12 — A/B Evaluation](docs/12-ab-evaluation-architecture.md) | Traffic-splitting evaluation design |
| [13 — Explainability Design](docs/13-explainability-design.md) | Routing explainability API design |
| [14 — Operational Events](docs/14-operational-events.md) | Structured event catalog |
| [15 — Metrics Specification](docs/15-metrics-specification.md) | Admin metrics contract |

## License

MIT

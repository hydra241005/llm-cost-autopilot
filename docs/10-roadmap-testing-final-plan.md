# 10 — Implementation Roadmap, Testing Strategy & Final Plan

## 1. Milestones

Estimates assume one focused engineer; total ≈ 78 h (~13 working days), matching the original 2-week envelope.

### M0 — Skeleton & tooling (4 h)
- **Files:** pyproject.toml, Makefile, ruff/mypy config, `src/autopilot` package skeleton, `config.py`, structlog setup, CI workflow, .env.example.
- **Deps:** none.
- **Output:** `make dev` boots an empty FastAPI app with `/health`; CI green.
- **Testing:** smoke test on /health; lint gate.

### M1 — Domain layer & model registry (6 h)
- **Files:** `domain/entities.py`, `enums.py`, `errors.py`, `interfaces.py`, `policies/pricing.py`, `infrastructure/providers/registry.py`, `configs/models.yaml`.
- **Output:** registry loads YAML → ModelConfig objects; pricing math (cost + counterfactual baseline) pure and tested.
- **Testing:** unit tests on pricing edge cases (zero tokens, free local model), registry validation errors.

### M2 — Provider adapters (8 h)
- **Files:** `providers/base.py`, three adapters, error mapping, `resilience/retry.py`.
- **Deps:** M1; provider keys; Ollama pulled.
- **Output:** same 10 prompts through all models via one interface; baseline comparison table saved as `data/baseline_run.json` (original Phase 1 deliverable, kept).
- **Testing:** unit tests with `respx`-mocked HTTP per adapter (success, 429, 500, timeout, malformed); one live smoke test per provider behind a marker.

### M3 — Persistence & migrations (6 h)
- **Files:** `persistence/db.py`, `models.py`, `repositories.py`, Alembic 0001+0002, `scripts/seed_db.py`.
- **Output:** schema live on Postgres and SQLite; repos pass round-trip tests.
- **Testing:** integration tests against dockerized Postgres AND SQLite (dual-dialect CI matrix).

### M4 — Classifier V1 (8 h)
- **Files:** `ml/features.py`, `ml/classifier.py`, `ml/model_store.py`, `ml/training/*`, `data/seed_prompts.jsonl` (200+ labeled), `data/holdout.jsonl`.
- **Output:** trained v1 artifact + metrics.json (target: ≥80% accuracy, Tier-3 recall ≥85%, calibration checked); HeuristicClassifier fallback.
- **Testing:** unit tests on every feature extractor; training determinism (fixed seed); holdout metrics asserted above floor in CI.

### M5 — Routing engine + API core (10 h)
- **Files:** `application/completion_orchestrator.py`, `policies/fallback.py`, `policies/escalation.py`, `resilience/circuit_breaker.py`, `api/*` (main, middleware, auth, rate_limit, schemas, routes/completions, routes/models), `configs/routing.yaml`.
- **Output:** end-to-end `POST /v1/completions` with classification, routing, retries, breaker, guardrails, metadata block; auth + rate limiting live.
- **Testing:** API tests with fake providers (deterministic); breaker state-machine unit tests; fallback-chain scenario tests; 429 header contract tests.

### M6 — Cache + queue + verification worker (10 h)
- **Files:** `cache/exact.py`, `queue/arq_client.py`, `worker/*`, `application/verification_service.py`, `policies/sampling.py`, `configs/rubrics/*`, judge implementation.
- **Output:** sampled async verification writing results; failures create training examples; sync guardrail escalation wired; cache hits logged as zero-cost rows.
- **Testing:** worker integration test with fake providers + real Redis; sampling policy unit tests (probabilistic branches with seeded RNG); judge JSON-parse repair path.

### M7 — Feedback retraining pipeline (6 h)
- **Files:** `worker/tasks.py::weekly_retrain`, `ml/training/evaluate.py` gate, admin retrain endpoints, hot-reload via pub/sub, canary + rollback.
- **Output:** end-to-end flywheel: injected failures → retrain → challenger promoted → API serves v2.
- **Testing:** integration test simulating 50 failures, asserting promotion gate behavior both directions (promote and reject).

### M8 — Stats API + remaining endpoints (5 h)
- **Files:** `stats_service.py`, `config_service.py`, routes stats/routing_config/admin, audit logging.
- **Output:** full API surface of doc 05.
- **Testing:** API contract tests; savings math cross-checked against hand-computed fixtures (verification step).

### M9 — Dashboard (10 h)
- **Files:** `dashboard/*` per doc 08 (theme.css, components, 7 pages).
- **Output:** premium dark dashboard with all states; screenshots for README.
- **Testing:** data-layer query unit tests; manual visual pass vs. doc 08 checklist; Lighthouse-style contrast check.

### M10 — Docker & deploy (5 h)
- **Files:** Dockerfiles, compose files, entrypoints, healthchecks, render.yaml, deploy notes.
- **Output:** `make up && make seed` → working stack from clean clone; one cloud deploy (Railway or Render) live.
- **Testing:** clean-machine boot test; failure sim: kill redis/postgres/ollama mid-run, verify degradation matches design.

### M11 — Load test, report & polish (≈ 8 h)
- **Files:** `scripts/loadtest.py` (1,000-prompt diverse corpus), `scripts/generate_report.py`, README with architecture diagram + headline number, case study md.
- **Output:** final savings report ("reduced cost X% at Y% quality parity, net of verification spend"), dashboard screenshots, demo GIF.
- **Testing:** the load test *is* the test; verify p95 router overhead < 10 ms, zero lost verification jobs, savings math reconciles: `Σsaved = Σbaseline − Σactual − Σverification`.

Dependency chain: M0 → M1 → {M2, M3} → M4 → M5 → M6 → M7 → {M8, M9} → M10 → M11. M2/M3 parallelizable; M8/M9 parallelizable.

## 2. Testing strategy (consolidated)

| Layer | Tooling | Scope & examples |
|---|---|---|
| Unit | pytest | domain policies (sampling probabilities with seeded RNG, escalation triggers, fallback ordering), pricing math, feature extraction (golden fixtures), breaker state machine, retry/backoff timing (frozen clock), judge output parsing |
| Integration | pytest + testcontainers (postgres, redis) | repositories round-trip both dialects, cache TTL/eviction, arq enqueue→execute, rate-limit Lua atomicity under concurrency, retrain promotion gate |
| API | httpx.AsyncClient + fake provider adapters | every endpoint's contract from doc 05: status codes, error envelope, validation messages, rate-limit headers, idempotency replay, auth scoping (admin vs normal) |
| Load | locust (`tests/load/`) + `scripts/loadtest.py` | 50 concurrent users, mixed-tier corpus; assert p95 latency budget, no 5xx at target RPS, queue drains |
| Failure simulation | compose-based chaos script | kill each dependency in turn: Redis down → cache/verification degrade but completions succeed; Postgres down → 503 with clear error; provider 500s → breaker opens → fallback chain observed; worker killed mid-job → arq redelivers |
| ML gates in CI | pytest | classifier holdout metrics above floor; calibration Brier ≤ threshold; training reproducible with fixed seed |

Coverage target: 85% on `domain` + `application` (the logic), pragmatic elsewhere. Every bug found during build gets a regression test — mention this discipline in the case study.

## 3. Additional production-grade features (impressive, not bloated)

1. **Idempotency keys** on completions (already in doc 05) — a real-payments-API habit.
2. **Shadow-mode config rollout:** a new routing config can run in "shadow" (decisions computed and logged, old config serves) before activation — trivially built on existing decision logging, huge interview signal.
3. **Per-key budgets:** monthly USD cap per API key with 402-style refusal — turns the project into a platform story.
4. **Drift monitor:** PSI (population stability index) on feature distributions week-over-week; alert when traffic shifts from training distribution.
5. **OpenAI-compatible endpoint alias** (`/v1/chat/completions` accepting the OpenAI schema) so any existing SDK can point at the autopilot with one base-URL change — the single best demo trick.
6. **Cost anomaly detection:** simple z-score on hourly spend with dashboard banner.
Deliberately excluded: Kubernetes, Kafka, feature stores, multi-tenant orgs/SSO — flagged in the README as "what I'd add at 100× scale" to show judgment rather than absence.

## 4. Final implementation plan (hand-off summary)

An engineer executes in this order: read docs 01–09 → M0 skeleton → M1 domain (pure logic first, everything else plugs into it) → M2 adapters + M3 persistence in parallel → M4 classifier (seed dataset writing is the long pole; start labeling early) → M5 the orchestrator (the heart — keep it a readable ~150-line use case that delegates to policies) → M6 async plane → M7 flywheel → M8/M9 surfaces → M10 packaging → M11 the portfolio run. At every milestone the definition of done is: tests listed above green, `make up` still boots, and the README quickstart still works from a clean clone.

The case study frame (write it last, from real numbers): *"I built an LLM routing layer that cut API costs 60%+ net of verification overhead, at 95%+ measured quality parity — with a feedback loop that retrains the router weekly from its own mistakes, champion/challenger gated, with automatic rollback."* Lead with the number; every claim in that sentence is backed by a table in Postgres.

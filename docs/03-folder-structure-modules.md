# 03 — Folder Structure & Module Design

Clean architecture (ports & adapters). The domain layer has zero framework imports; infrastructure implements domain interfaces; FastAPI and Streamlit are thin shells.

## 1. Repository layout

```
llm-cost-autopilot/
├── README.md                     # headline savings number, arch diagram, quickstart
├── docs/                         # this blueprint (01–10)
├── docker-compose.yml
├── docker-compose.prod.yml
├── .env.example
├── Makefile                      # make dev / test / seed / loadtest / retrain
├── pyproject.toml                # uv/poetry; ruff + mypy config
│
├── src/autopilot/
│   ├── __init__.py
│   ├── config.py                 # pydantic-settings: env-driven AppSettings
│   │
│   ├── domain/                   # ── pure Python, no I/O, no framework ──
│   │   ├── entities.py           # RequestRecord, Usage, RoutingDecision,
│   │   │                         # VerificationResult, EscalationEvent, ModelConfig
│   │   ├── enums.py              # Tier, Provider, RequestStatus, VerdictType
│   │   ├── interfaces.py         # abstract ports (see §3)
│   │   ├── errors.py             # domain exceptions hierarchy
│   │   └── policies/
│   │       ├── sampling.py       # SamplingPolicy (verify or not)
│   │       ├── escalation.py     # EscalationPolicy (sync guardrail rules)
│   │       ├── fallback.py       # FallbackChain
│   │       └── pricing.py        # cost math + counterfactual baseline
│   │
│   ├── application/              # ── use cases / orchestration ──
│   │   ├── completion_orchestrator.py   # the request lifecycle (doc 02 §3)
│   │   ├── stats_service.py             # savings, distributions, escalation rates
│   │   ├── config_service.py            # routing-config CRUD + versioning
│   │   └── verification_service.py      # shadow call + judge + record (used by worker)
│   │
│   ├── infrastructure/           # ── adapters implementing domain ports ──
│   │   ├── providers/
│   │   │   ├── base.py           # shared HTTP client, token counting, error mapping
│   │   │   ├── openai_adapter.py
│   │   │   ├── anthropic_adapter.py
│   │   │   ├── ollama_adapter.py
│   │   │   └── registry.py       # ModelRegistry: YAML → ModelConfig objects
│   │   ├── ml/
│   │   │   ├── features.py       # FeatureExtractor (doc 06 §6)
│   │   │   ├── classifier.py     # SklearnClassifier implements ComplexityClassifier
│   │   │   ├── model_store.py    # versioned artifact load/save, hot reload
│   │   │   └── training/
│   │   │       ├── dataset.py    # build/dedupe/balance dataset
│   │   │       ├── train.py      # challenger training
│   │   │       └── evaluate.py   # champion vs challenger gate
│   │   ├── persistence/
│   │   │   ├── db.py             # async engine/session factory
│   │   │   ├── models.py         # SQLAlchemy ORM (doc 04)
│   │   │   ├── repositories.py   # RequestRepository etc. implementations
│   │   │   └── migrations/       # Alembic
│   │   ├── cache/
│   │   │   ├── exact.py          # RedisExactCache
│   │   │   └── semantic.py       # SemanticCache (feature-flagged)
│   │   ├── resilience/
│   │   │   ├── circuit_breaker.py
│   │   │   └── retry.py
│   │   ├── queue/
│   │   │   └── arq_client.py     # enqueue + job definitions
│   │   └── observability/
│   │       ├── logging.py        # structlog JSON config, request_id contextvars
│   │       ├── metrics.py        # prometheus counters/histograms
│   │       └── tracing.py        # optional OTel
│   │
│   ├── api/                      # ── FastAPI shell ──
│   │   ├── main.py               # app factory, lifespan (load classifier, warm registry)
│   │   ├── dependencies.py       # DI wiring: settings → adapters → use cases
│   │   ├── middleware.py         # request_id, timing, error envelope
│   │   ├── auth.py               # API-key resolution + hashing
│   │   ├── rate_limit.py         # Redis token bucket
│   │   ├── schemas/              # Pydantic request/response models (doc 05)
│   │   └── routes/
│   │       ├── completions.py    # POST /v1/completions
│   │       ├── models.py         # GET /v1/models
│   │       ├── stats.py          # GET /v1/stats*
│   │       ├── routing_config.py # GET/PUT /v1/routing-config
│   │       └── admin.py          # health, retrain trigger, keys
│   │
│   └── worker/
│       ├── main.py               # arq WorkerSettings
│       └── tasks.py              # verify_request, weekly_retrain
│
├── dashboard/                    # Streamlit app (doc 08)
│   ├── app.py
│   ├── pages/                    # 1_analytics.py … 7_settings.py
│   ├── components/               # cards, charts, tables, states
│   ├── styles/theme.css          # design tokens + glassmorphism
│   └── data.py                   # read-only queries against Postgres
│
├── configs/
│   ├── models.yaml               # model registry: pricing, tiers, limits
│   ├── routing.yaml              # default tier→model map + fallbacks
│   └── rubrics/                  # judge rubrics per task type (md/yaml)
│
├── data/
│   ├── seed_prompts.jsonl        # 200+ labeled prompts (tier labels)
│   └── holdout.jsonl             # frozen eval set — never trained on
│
├── scripts/
│   ├── seed_db.py
│   ├── loadtest.py               # drives 1,000-prompt portfolio run
│   └── generate_report.py        # final savings report (md + charts)
│
└── tests/
    ├── unit/                     # domain policies, features, pricing, breaker
    ├── integration/              # repos vs real Postgres, cache vs real Redis
    ├── api/                      # httpx AsyncClient against app w/ fake providers
    ├── load/                     # locust file
    └── fixtures/                 # canned provider responses, fake adapters
```

## 2. Package responsibilities & dependency flow

```
api ──▶ application ──▶ domain ◀── infrastructure ◀── worker
dashboard ──▶ (Postgres read-only; no imports from src except schemas/pricing)
```

Rules enforced by convention + an import-linter contract in CI:

- `domain` imports only stdlib + pydantic (for value objects). Never sqlalchemy/fastapi/redis/sklearn.
- `application` imports `domain` only; receives adapters through constructor injection typed as domain interfaces.
- `infrastructure` imports `domain` (to implement its interfaces) — never `application` or `api`.
- `api`/`worker` are composition roots: the only places where concrete infrastructure classes are constructed and wired into use cases (`dependencies.py`, `worker/main.py`).

## 3. Domain interfaces (ports)

| Interface | Key methods | Implemented by |
|---|---|---|
| `LLMProvider` (ABC) | `async complete(prompt, params) -> ProviderResponse`; `count_tokens(text)`; `provider_name` | OpenAI/Anthropic/Ollama adapters |
| `ComplexityClassifier` (ABC) | `predict(features) -> (Tier, confidence)`; `version` | `SklearnClassifier`; `HeuristicClassifier` (cold-start fallback) |
| `FeatureExtractor` (Protocol) | `extract(prompt, params) -> FeatureVector` | `ml/features.py` |
| `ResponseCache` (ABC) | `get(key)`, `set(key, value, ttl)`, `make_key(prompt, params)` | exact + semantic caches |
| `RequestRepository` (ABC) | `save(record)`, `get(request_id)`, `stats_window(...)` | SQLAlchemy repo |
| `VerificationRepository` (ABC) | `save(result)`, `failure_rate(model, window)` | SQLAlchemy repo |
| `ConfigRepository` (ABC) | `active_routing_config()`, `save_version(cfg, actor)` | SQLAlchemy repo |
| `JobQueue` (ABC) | `enqueue_verification(payload)`, `enqueue_retrain()` | arq client |
| `Judge` (ABC) | `async evaluate(task_type, prompt, candidate, reference) -> JudgeVerdict` | LLM judge via provider port |
| `RateLimiter` (ABC) | `async allow(key) -> RateDecision` | Redis token bucket |
| `Clock`, `IdGenerator` | trivial | real + deterministic test doubles |

Why ABCs and not just duck typing: every port has at least two implementations (real + test fake), and interviewers read this as testability by design.

## 4. Key abstract class sketch (signatures only)

```
class LLMProvider(ABC):
    name: Provider
    @abstractmethod
    async def complete(self, req: CompletionInput, cfg: ModelConfig,
                       timeout_s: float) -> ProviderResponse: ...
    # Raises: ProviderTimeout, ProviderRateLimited, ProviderServerError,
    #         ProviderBadRequest (mapped from vendor-specific exceptions)

class ComplexityClassifier(ABC):
    version: str
    @abstractmethod
    def predict(self, fv: FeatureVector) -> Prediction: ...  # tier + confidence + per-class proba
```

Error mapping is a first-class design point: each adapter translates vendor exceptions into the shared hierarchy in `domain/errors.py`, so the retry/breaker logic is provider-agnostic.

## 5. Configuration strategy

- `AppSettings` (pydantic-settings) reads env once at startup; injected everywhere — no `os.environ` calls scattered in code.
- `configs/models.yaml` and `configs/routing.yaml` are the *seed* config; on first boot they are imported into Postgres (`routing_configs`), after which the DB is the source of truth and the PUT endpoint creates new versions. YAML stays in git as reviewable defaults.
- Feature flags (semantic cache, tracing, judge model choice) are plain settings — no flag service needed at this scale.

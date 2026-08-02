# 02 — System Architecture & Diagrams

All diagrams are Mermaid; they render on GitHub and in most IDEs.

## 1. High-level architecture

```mermaid
flowchart LR
    subgraph clients [Clients]
        C[SDK / curl / Playground]
    end

    subgraph api [API Service - FastAPI]
        MW[Auth + Rate Limit + Validation]
        CACHE{Cache lookup}
        CLS[Complexity Classifier]
        RTR[Routing Engine]
        GRD[Guardrail Check]
    end

    subgraph providers [Provider Adapters]
        OAI[OpenAI]
        ANT[Anthropic]
        OLL[Ollama local]
    end

    subgraph asyncplane [Async Plane]
        Q[(Redis - arq queue)]
        WRK[Verification Worker]
        JUDGE[LLM-as-Judge]
        TRAIN[Weekly Retrainer]
    end

    subgraph data [Data Plane]
        PG[(PostgreSQL)]
        RD[(Redis - cache and rate limits)]
        ART[Classifier Artifacts v1..vN]
    end

    DASH[Streamlit Dashboard]

    C --> MW --> CACHE
    CACHE -- hit --> C
    CACHE -- miss --> CLS --> RTR --> OAI & ANT & OLL
    OAI & ANT & OLL --> GRD --> C
    GRD -- hard fail --> RTR
    GRD -- sampled --> Q --> WRK --> JUDGE
    WRK --> PG
    MW --> RD
    CACHE --> RD
    RTR --> PG
    TRAIN --> PG
    TRAIN --> ART --> CLS
    DASH --> PG
```

**Services (containers):** `api` (FastAPI), `worker` (arq), `retrainer` (cron-style job, can share worker image), `postgres`, `redis`, `ollama`, `dashboard` (Streamlit).

## 2. Component architecture (inside the API service)

```mermaid
flowchart TD
    subgraph presentation [Presentation Layer]
        EP[api/routes/*  - completions, models, stats, config, admin]
        DEPS[api/dependencies - auth, rate limit, request context]
    end
    subgraph application [Application Layer]
        UC1[CompletionOrchestrator]
        UC2[StatsService]
        UC3[ConfigService]
    end
    subgraph domain [Domain Layer - pure, no I/O]
        D1[entities: RequestRecord, RoutingDecision, VerificationResult]
        D2[interfaces: LLMProvider, Classifier, Cache, RequestRepository]
        D3[policies: SamplingPolicy, EscalationPolicy, FallbackChain]
    end
    subgraph infrastructure [Infrastructure Layer]
        I1[providers: OpenAIAdapter, AnthropicAdapter, OllamaAdapter]
        I2[ml: SklearnClassifier, FeatureExtractor, ModelStore]
        I3[persistence: SQLAlchemy repos, Alembic]
        I4[cache: RedisExactCache, SemanticCache]
        I5[resilience: CircuitBreaker, RetryPolicy]
        I6[queue: ArqEnqueuer]
    end
    EP --> UC1 & UC2 & UC3
    UC1 --> D2 & D3
    I1 & I2 & I3 & I4 & I6 -. implement .-> D2
    UC1 -.uses via interface.-> I1 & I2 & I3 & I4 & I6
    I1 --> I5
```

Dependency rule: **presentation → application → domain ← infrastructure**. The domain layer imports nothing from the outer layers; infrastructure implements domain interfaces (ports & adapters).

## 3. Request lifecycle

1. **Ingress** — request hits `POST /v1/completions`; middleware assigns `request_id` (UUIDv7), resolves API key → tenant, checks token-bucket rate limit in Redis (`429` on exceed), validates body via Pydantic (`422` on failure), enforces max prompt size.
2. **Cache** — normalized prompt+params hashed; exact-match Redis lookup. Hit → return cached response with `"cached": true` metadata, log a zero-cost row, done (~2 ms).
3. **Classification** — FeatureExtractor computes features (<1 ms); versioned sklearn classifier returns `(tier, confidence)`. If `confidence < τ` (default 0.6), bump one tier and flag `low_confidence=true`.
4. **Routing** — RoutingEngine reads the active tier→model map; skips models whose circuit breaker is open; produces an ordered candidate list `[primary, fallback1, fallback2]`.
5. **Execution** — provider adapter called with per-tier timeout; retryable errors (429/5xx/timeout) retried with exponential backoff + jitter (max 2); on exhaustion, next candidate in the fallback chain; breaker records outcome.
6. **Guardrails (sync)** — cheap checks on the output: empty, refusal pattern, truncated (`finish_reason=length`), requested-format violation. Hard failure → immediate synchronous escalation to next tier (once), `escalated=true`.
7. **Response** — standardized payload + `routing` metadata block (model, tier, confidence, cost, latency, cache/escalation flags) returned to the client. Total added overhead budget: **< 10 ms** over the raw provider call.
8. **Persistence & enqueue (post-response)** — request row written to Postgres; SamplingPolicy decides whether to enqueue a verification job in arq (see §5).
9. **Verification (async)** — worker shadow-calls the reference model, runs LLM-as-judge, writes `verification_results`; failures create `training_examples`.
10. **Learning (weekly)** — retrainer builds a dataset from labels + accumulated failures, trains a challenger, evaluates vs champion on frozen holdout, promotes or rejects, bumps classifier version.

## 4. Sequence diagram — happy path + async verification

```mermaid
sequenceDiagram
    autonumber
    participant CL as Client
    participant API as FastAPI
    participant RD as Redis
    participant CF as Classifier
    participant PR as Provider (Haiku)
    participant PG as Postgres
    participant W as Worker
    participant REF as Reference model (GPT-4o)
    participant J as Judge

    CL->>API: POST /v1/completions
    API->>RD: auth key check + rate limit + cache lookup
    RD-->>API: miss
    API->>CF: extract features, predict
    CF-->>API: tier=1, confidence=0.91
    API->>PR: send_request (timeout=10s)
    PR-->>API: output, usage
    API->>API: guardrail check → pass
    API-->>CL: 200 + routing metadata
    API->>PG: insert request row
    API->>RD: cache set + enqueue verify (sampled)
    RD->>W: dequeue job
    W->>REF: shadow call (same prompt)
    W->>J: judge(cheap_output, ref_output, rubric)
    J-->>W: score 4.6 / agreement high
    W->>PG: insert verification_result (pass)
```

## 5. Async verification flow (with failure branch)

```mermaid
flowchart TD
    A[Response delivered] --> B{SamplingPolicy}
    B -- skip --> Z1[No job]
    B -- sample --> C[Enqueue arq job]
    C --> D[Worker: shadow call reference model]
    D -- provider error --> D2[Retry x3 backoff] --> D
    D --> E[LLM-as-judge: rubric score + pairwise agreement]
    E --> F{score >= threshold?}
    F -- yes --> G[Log PASS verification_result]
    F -- no --> H[Log FAIL + quality gap]
    H --> I[Create training_example: prompt features -> corrected tier]
    H --> J[Increment failure-rate metric per tier/model]
    J --> K{failure rate > alert threshold?}
    K -- yes --> L[Alert + optionally auto-raise tier mapping]
```

Sampling policy: `P(verify) = 1.0` if `low_confidence` or inside canary window (config/classifier changed < N hours ago), else `base_rate` (default 0.10). Daily verification budget cap in USD; when exhausted, only low-confidence requests are verified.

## 6. Feedback / retraining pipeline

```mermaid
flowchart LR
    A[(training_examples\nseed labels + failures)] --> B[Weekly job]
    B --> C[Build dataset\ndedupe, class-balance]
    C --> D[Train challenger vN+1]
    D --> E[Evaluate on frozen holdout\naccuracy, per-class F1,\ncost-weighted error]
    E --> F{beats champion\non cost-weighted error?}
    F -- yes --> G[Promote: write artifact vN+1\n+ metrics.json, update registry]
    F -- no --> H[Reject + report]
    G --> I[API hot-reloads classifier\nnext prediction uses vN+1]
    G --> J[Canary window: verify 100%\nfor next 24h]
```

## 7. Logging pipeline

```mermaid
flowchart LR
    A[API structlog JSON] --> ST[stdout]
    B[Worker structlog JSON] --> ST
    ST --> DC[docker logs / promtail optional]
    A2[Request rows] --> PG[(Postgres: requests,\nrouting_decisions,\nverification_results,\nescalations, audit_log)]
    M[Prometheus /metrics] --> PR2[Prometheus optional]
    PG --> DASH[Streamlit dashboard]
```

Every log line and DB row carries `request_id`; the worker receives it inside the job payload so one grep or one SQL query reconstructs the full lifecycle of any request.

## 8. Data flow summary

| Data | Producer | Store | Consumer |
|---|---|---|---|
| Request/response metadata | API | Postgres `requests` | Dashboard, stats API |
| Routing decision + classifier version | Router | `routing_decisions` | Retrainer, dashboard |
| Verification scores | Worker | `verification_results` | Dashboard, alerting, retrainer |
| Training examples | Worker + seed set | `training_examples` | Weekly retrainer |
| Classifier artifacts | Retrainer | volume `artifacts/` + `classifier_versions` | API classifier |
| Cache entries | API | Redis (TTL) | API |
| Rate-limit counters | Middleware | Redis | Middleware |
| Config (tier→model map) | Admin API | `routing_configs` (versioned rows) | Router |

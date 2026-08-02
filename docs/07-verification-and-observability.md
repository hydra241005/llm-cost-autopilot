# 07 — Quality Verification & Logging/Observability

## Part A — Quality Verification System

### A1. What gets verified (sampling policy)

Verifying everything would erase the savings (doc 01, W3). Policy, evaluated post-response:

| Condition | P(verify) |
|---|---|
| `low_confidence` flag on routing decision | 1.0 |
| Inside canary window (config/classifier changed < 24 h) | 1.0 |
| Sync escalation occurred | 1.0 |
| Otherwise | `base_rate` = 0.10 |
| Daily verification budget (USD) exhausted | only low-confidence |

All knobs live in the routing config; every verification row stores `sampled_reason` so the dashboard can show unbiased vs. targeted quality estimates separately (targeted samples are *not* mixed into the headline pass rate — selection bias).

### A2. Verification job (worker)

1. Load request (features, prompt if stored; if prompt not stored, verification is skipped and the sample is re-drawn from storing keys — documented limitation).
2. **Shadow call** the reference model (Tier-3 primary) with identical params.
3. **Judge** both outputs (below).
4. Persist `verification_results` (score, agreement, verdict, quality_gap, rationale, shadow cost).
5. On FAIL: create `training_example` (features → corrected tier), increment per-model failure-rate metric.
6. Retries: 3 with backoff on provider errors; terminal failure logs `verdict=judge_error` (never silently dropped — the flywheel depends on it).

### A3. LLM-as-judge design

**Judge model:** mid-tier by default (gpt-4o-mini / haiku) — cheap and adequate for rubric scoring; configurable.
**Rubrics per task type** (`configs/rubrics/*.md`):

| Task type | Rubric axes (each 1–5) |
|---|---|
| extraction | field completeness, field accuracy, format compliance |
| summarization | faithfulness (no hallucinated claims), coverage of key points, concision |
| classification | label agreement with reference, justification validity |
| generation | instruction adherence, coherence, completeness |
| reasoning | logical validity, final-answer agreement, step soundness |

**Bias controls:** pairwise comparisons are run twice with positions swapped (A/B then B/A); disagreement between the two orderings ⇒ verdict `judge_error` rather than a coin flip. Judge must output structured JSON (`score`, `agreement`, `rationale`) — parse failures retry once with a repair prompt. Verbosity bias mitigated by rubric instruction ("do not reward length").
**Calibration:** a 30-item human-labeled calibration set is scored by the judge monthly; if Spearman correlation with human scores drops below 0.7, alert and freeze auto-promotion.

### A4. Thresholds, disagreement, escalation policy

- PASS: `judge_score ≥ 4.0` AND `agreement ≥ 0.7` (both configurable per task type).
- FAIL: below either → training example + failure metric.
- **Disagreement detection at the fleet level:** per (tier, model) rolling failure rate over 24 h; if > 2× the 7-day baseline or > 8% absolute → alert; if sustained 48 h → recommended action surfaced on dashboard: "raise Tier N primary to next model" (one-click applies a new routing config version — human approves, system suggests).
- **Sync escalation** (pre-response) is separate and heuristic-only: empty output, refusal patterns on non-refusable tasks, `finish_reason=length` truncation, format violation when a format was requested. One escalation max per request.

### A5. Weekly retraining pipeline

Covered in doc 06 §B5 — verification failures are its fuel; champion/challenger gate + canary + auto-rollback close the loop safely.

---

## Part B — Logging & Observability

### B1. Structured logging

- **structlog** with a JSON renderer; every line: `timestamp, level, event, request_id, service, version` + event-specific fields.
- `request_id` (UUIDv7) generated in middleware, stored in a contextvar so every log call in the request scope inherits it; passed inside arq job payloads so worker logs correlate.
- Canonical events (stable names, greppable): `request_received`, `cache_hit`, `classified`, `routed`, `provider_call`, `provider_retry`, `breaker_open`, `guardrail_escalation`, `response_sent`, `verify_enqueued`, `verify_completed`, `retrain_started`, `retrain_promoted`, `config_updated`.
- **One canonical log line per request** (`response_sent`) carrying the whole summary (tier, model, cost, latency, cache, escalation) — the Stripe-style pattern that makes ad-hoc analysis trivial.
- Redaction: prompts never appear in logs (only hashes + lengths); provider keys masked by a structlog processor.

### B2. Audit trail

Two layers: the `requests`/`routing_decisions`/`verification_results`/`escalation_events` tables are the per-request audit trail (queryable, joinable, survives log rotation); `audit_log` records human/admin actions (config changes with before/after diff, key lifecycle, manual promotions).

### B3. Metrics (Prometheus `/metrics`)

| Metric | Type | Labels |
|---|---|---|
| `requests_total` | counter | status, tier, model, cache_hit |
| `request_latency_seconds` | histogram | tier, model |
| `provider_latency_seconds` | histogram | provider, model |
| `cost_usd_total` / `baseline_cost_usd_total` | counter | model |
| `escalations_total` | counter | kind, trigger |
| `verification_verdicts_total` | counter | verdict, tier, model |
| `circuit_breaker_state` | gauge | model |
| `queue_depth` | gauge | queue |
| `classifier_confidence` | histogram | version |

Savings% and escalation-rate are derived in Grafana/dashboard, not pre-computed (metrics stay raw).

### B4. Tracing (optional, feature-flagged)

OpenTelemetry with FastAPI + httpx auto-instrumentation; spans: `classify → route → provider_call(×n) → guardrail`; worker continues the trace via propagated context in the job payload. Exporter: OTLP → Jaeger container in the `docker-compose.prod.yml` observability profile. Off by default to keep the core compose light.

### B5. Alerting (lightweight, portfolio-appropriate)

A worker-side `health_sweep` cron every 5 min evaluates: breaker open > 5 min, verification failure-rate spike, daily cost > budget, queue depth > 500, judge calibration stale. Alerts write an `alerts` row surfaced as a dashboard banner and (optional) webhook (Slack/Discord URL via env). No PagerDuty — right-sized for the project, and the *design* shows you know what real alerting is for.

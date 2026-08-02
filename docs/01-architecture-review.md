# 01 — Critical Architecture Review

**Project:** LLM Cost Autopilot — an intelligent routing layer that classifies request complexity, routes to the cheapest capable model, and continuously verifies routing quality.

This document reviews the original 6-phase specification, identifies its weaknesses, and states every design change made in this blueprint with justification. All downstream documents (02–10) assume the **revised** architecture.

---

## 1. Verdict on the original spec

The original spec is a solid *learning* project but has gaps that a senior interviewer will probe in the first ten minutes. The core idea (classify → route → verify → feed back) is sound and is preserved intact. The problems are in the operational layer: storage, queuing, failure handling, security, and evaluation rigor.

## 2. Identified weaknesses and fixes

### W1 — SQLite as the production database
**Problem.** SQLite allows a single writer. The API writes request logs, the verifier writes quality scores, and the retraining job reads training data — three concurrent writers/readers. Under load-test conditions (Phase 6: 500–1,000 prompts) you will hit `database is locked` errors, and the interviewer question "what happens at 100 RPS?" has no good answer.
**Fix.** SQLAlchemy 2.0 + Alembic with **PostgreSQL in Docker/production and SQLite for zero-dependency local dev**. Same ORM models, one env var to switch. This costs almost nothing and converts "toy" into "production posture." (Decision confirmed with project owner.)

### W2 — No message queue for async verification
**Problem.** The spec says "queue an async job" but names no queue. `asyncio.create_task` inside the API process loses jobs on restart, can't be scaled independently, and offers no retry semantics — verification is the flywheel of the whole system, so losing jobs silently corrupts the training signal.
**Fix.** **Redis + arq** (async-native, lightweight, same author as pydantic). Verification jobs are persisted, retried with backoff, and processed by a separate worker container. Redis also serves the response cache and rate-limit counters, so it earns its place three times over.

### W3 — Verifying 100% of requests destroys the economics
**Problem.** The spec sends *every* request to the highest-tier model for verification. If every Haiku call triggers a GPT-4o shadow call, the system **increases** total cost — the opposite of its purpose. This is the single biggest logical flaw in the spec and the first thing a sharp interviewer will catch.
**Fix.** **Sampled verification**: verify 100% of low-classifier-confidence requests, ~10% random sample of high-confidence ones (configurable), and 100% of requests in the first N hours after a routing-config or classifier change ("canary window"). This preserves the statistical feedback signal at ~10–15% of the verification cost. The sampling policy itself becomes a talking point.

### W4 — Auto-escalation "if latency permits" is underspecified
**Problem.** The spec suggests re-running with the bigger model and "returning the better result," but the response has already been returned to the user — you can't retroactively swap it in a synchronous API.
**Fix.** Split escalation into two explicit modes: (a) **synchronous guardrail escalation** — cheap pre-response heuristics (empty output, refusal detection, truncation, format-violation) trigger an immediate re-route *before* returning; (b) **asynchronous learning escalation** — verifier failures never change a delivered response; they are logged, counted, and fed to retraining. This distinction is architecturally honest and interview-defensible.

### W5 — No failure handling for providers
**Problem.** No retries, timeouts, circuit breakers, or fallbacks. Provider outages (Anthropic/OpenAI both have incidents monthly) would 500 the API.
**Fix.** Per-provider **circuit breaker** (closed → open → half-open), retries with exponential backoff + jitter on retryable errors only (429/5xx/timeouts), per-tier timeout budgets, and a **fallback chain** per tier (e.g., Tier 2: gpt-4o-mini → claude-haiku → ollama). Detailed in doc 06.

### W6 — No security model
**Problem.** An open proxy that spends your OpenAI budget is a security incident, not a portfolio piece. No authN, no rate limiting, no protection against prompt logging leaks.
**Fix.** API-key auth (keys stored as SHA-256 hashes, prefix-identifiable like `lca_live_...`), per-key token-bucket rate limiting in Redis, request-size caps, provider keys only via env/secret manager, and **prompt storage policy**: full prompts stored only when `store_prompt=true` per key; otherwise only a hash + extracted features (needed for retraining while respecting data sensitivity).

### W7 — Classifier evaluation is too thin
**Problem.** "80% accuracy" on a hand-built dataset is a weak claim: classes are imbalanced, and the *cost* of errors is asymmetric (routing Tier 3 → Tier 1 hurts quality; Tier 1 → Tier 3 only wastes money).
**Fix.** Report per-class precision/recall + confusion matrix, and introduce a **cost-weighted error metric** (under-routing penalized more than over-routing). Add a **confidence threshold**: below it, the router bumps up one tier ("when unsure, spend a little more"). Full design in doc 06.

### W8 — LLM-as-judge used naively
**Problem.** Comparing outputs to "what GPT-4o would say" measures agreement, not quality; judges are biased toward verbosity and their own outputs.
**Fix.** Rubric-based judging (task-type-specific rubrics: extraction completeness, summary faithfulness, instruction adherence), score 1–5 with required justification, position-swapped pairwise comparison when comparing outputs, and periodic judge calibration against a small human-labeled set. Doc 07.

### W9 — No caching
**Problem.** The cheapest LLM call is the one you never make. A cost-optimization project without caching is incomplete.
**Fix.** Two-level cache in Redis: exact-match (SHA-256 of normalized prompt + params, short TTL) and optional **semantic cache** (embedding similarity ≥ 0.97, feature-flagged) with cache-hit savings surfaced on the dashboard as a separate line item.

### W10 — Retraining pipeline lacks safety
**Problem.** "Retrain weekly on failures" with no versioning, no holdout, no rollback — a bad retrain silently degrades routing for a week.
**Fix.** Versioned classifier artifacts (`classifier/v{n}/model.joblib` + metrics.json), champion/challenger evaluation on a frozen holdout set before promotion, automatic rollback if the challenger underperforms, and every routing decision logs the classifier version that made it. Doc 06.

### W11 — Observability is just "logs"
**Problem.** Structured logs alone can't answer "why was p99 latency high at 3pm?"
**Fix.** Three pillars: structlog JSON logs with a `request_id` correlation ID propagated to the worker, Prometheus metrics (`/metrics`), and optional OpenTelemetry tracing. Doc 07.

### W12 — Cost baseline claim needs rigor
**Problem.** "Saved X% vs GPT-4o for everything" is computed against a hypothetical. If tokenization differs across models, the counterfactual is wrong.
**Fix.** Compute the counterfactual using the *actual* input tokens and the premium model's *measured* output-length distribution for the same task tier (or, during verification samples, the real shadow-call cost). Show both "estimated" and "measured-on-sample" savings — this honesty is a differentiator.

## 3. What is intentionally NOT changed

- **Scikit-learn classifier for V1.** A fine-tuned transformer would be slower, costlier, and harder to explain. A logistic regression with good features that gets retrained on real failure data is the *senior* answer: simplest thing that closes the loop.
- **Streamlit for the dashboard.** Grafana is ops-only; a custom React app doubles the project. Streamlit with heavy custom CSS achieves the premium look at 20% of the effort (doc 08).
- **Three complexity tiers.** More tiers = sparser training data per tier. Three is right for V1; the schema supports N tiers for the future.
- **The 2-week scope.** Every improvement above fits the original timeline (doc 10 has the revised hour estimates).

## 4. Revised architecture summary

```
Client ──▶ FastAPI (auth, rate-limit, validate)
              │
              ├─▶ Cache (Redis: exact + semantic) ── hit ──▶ return
              │
              ├─▶ Complexity Classifier (features + sklearn, versioned)
              │        │ confidence < τ → bump tier
              ├─▶ Router (tier→model map, circuit breakers, fallback chain)
              ├─▶ Provider Adapters (OpenAI / Anthropic / Ollama)
              │        │ guardrail check → sync escalation if hard-fail
              ├─▶ Response to client (with routing metadata)
              │
              └─▶ enqueue verification job (sampled) ──▶ Redis (arq)
                                                          │
                                          Worker: shadow call → LLM-judge →
                                          score → log verification →
                                          failure ⇒ training_example row
                                                          │
                                          Weekly retrain: champion/challenger
                                          → new classifier version → Router
```

Full diagrams in doc 02.

## 5. Risk register (top 5)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Provider API changes / outages | Medium | High | Adapter layer, circuit breaker, fallback chain |
| Classifier drift as traffic changes | High | Medium | Weekly retrain, drift metric on feature distributions |
| Verification cost blowing the budget | Medium | High | Sampling policy, per-day verification budget cap |
| Judge bias inflating quality scores | Medium | Medium | Rubrics, position swap, human calibration set |
| Demo data too small to be convincing | High | Medium | Load-test corpus of 1,000 diverse prompts (doc 10) |

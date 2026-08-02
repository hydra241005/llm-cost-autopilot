# 05 — API Specification

Base URL: `/v1`. Auth: `Authorization: Bearer lca_live_...` on every endpoint except `/health`. All errors use one envelope:

```json
{ "error": { "type": "rate_limit_exceeded", "message": "…", "request_id": "…", "param": null } }
```

Error types: `invalid_request` (422), `authentication_error` (401), `permission_denied` (403), `not_found` (404), `rate_limit_exceeded` (429), `provider_unavailable` (503), `internal_error` (500), `payload_too_large` (413).

---

## 1. POST /v1/completions — the product

**Request**

```json
{
  "messages": [{"role": "user", "content": "Summarize this article: …"}],
  "max_tokens": 1024,
  "temperature": 0.7,
  "task_type": "summarization",
  "routing": { "force_tier": null, "allow_cache": true, "metadata": {"trace": "checkout-42"} }
}
```

| Field | Type | Validation |
|---|---|---|
| messages | array, required | 1–50 items; role ∈ {system,user,assistant}; total content ≤ 128 KB (else 413) |
| max_tokens | int, optional | 1–8192, default 1024 |
| temperature | float, optional | 0–2, default 0.7 |
| task_type | enum, optional | extraction / summarization / classification / generation / reasoning / other — improves rubric choice; inferred if absent |
| routing.force_tier | int/null | 1–3; admin-scoped keys only (403 otherwise) — for debugging |
| routing.allow_cache | bool | default true |

**Response 200**

```json
{
  "id": "req_0190f3…",
  "object": "completion",
  "created": 1753948800,
  "content": "…model output…",
  "finish_reason": "stop",
  "usage": { "input_tokens": 812, "output_tokens": 214 },
  "routing": {
    "model": "anthropic:claude-haiku",
    "provider": "anthropic",
    "tier_predicted": 1,
    "tier_effective": 1,
    "confidence": 0.91,
    "classifier_version": "v3",
    "cache_hit": false,
    "escalated": false,
    "fallback_depth": 0,
    "cost_usd": 0.000412,
    "baseline_cost_usd": 0.004890,
    "saved_usd": 0.004478,
    "latency_ms": 1240
  }
}
```

**Errors:** 422 (validation, includes `param`), 401, 429 (with `Retry-After` header), 503 when the full fallback chain is exhausted (breaker states included in detail for admin keys), 413 oversize.

**Status codes:** 200, 401, 403, 413, 422, 429, 500, 503.

---

## 2. GET /v1/models

Returns registry with live health.

```json
{ "data": [ {
    "id": "openai:gpt-4o-mini", "provider": "openai", "quality_tier": 2,
    "input_cost_per_mtok": 0.15, "output_cost_per_mtok": 0.60,
    "avg_latency_ms": 900, "is_active": true, "circuit_state": "closed" } ] }
```

Query params: `?active=true`, `?tier=2`. Codes: 200, 401.

## 3. GET /v1/stats

Query params: `window` (`24h`/`7d`/`30d`/`all`, default `7d`), `group_by` (`day`/`hour`).

```json
{
  "window": "7d",
  "totals": {
    "requests": 4210, "cost_usd": 6.91, "baseline_cost_usd": 19.44,
    "saved_usd": 12.53, "savings_pct": 64.4,
    "cache_hits": 512, "cache_saved_usd": 1.10,
    "verification_cost_usd": 0.83, "net_savings_pct": 60.2
  },
  "routing_distribution": [ {"model": "anthropic:claude-haiku", "share": 0.46, "requests": 1937} ],
  "quality": { "avg_judge_score": 4.42, "pass_rate": 0.958, "sampled": 431 },
  "escalations": { "sync": 23, "async_flagged": 31, "rate": 0.0128 },
  "series": [ {"date": "2026-07-30", "cost_usd": 0.92, "baseline_cost_usd": 2.71, "requests": 603} ]
}
```

Note `net_savings_pct`: savings *after* subtracting verification spend — the honest headline number. Codes: 200, 401, 422 (bad window).

## 4. GET /v1/stats/requests — logs explorer feed

Params: `limit` (≤200), `cursor` (keyset pagination on UUIDv7), `model`, `tier`, `verdict`, `escalated`, `since`, `until`. Returns request rows joined with decision + verification. Codes: 200, 401, 422.

## 5. GET /v1/requests/{request_id}

Full lifecycle of one request: request row, routing decision, verification result, escalation events, timeline. 404 if unknown; 403 if owned by another key (non-admin).

## 6. GET /v1/routing-config

Returns the active config version:

```json
{ "id": 7, "created_at": "…", "created_by": "lca_live_ab12", "comment": "haiku for tier2 trial",
  "config": { "tiers": {
      "1": { "primary": "ollama:llama3.1-8b", "fallbacks": ["anthropic:claude-haiku"] },
      "2": { "primary": "openai:gpt-4o-mini", "fallbacks": ["anthropic:claude-sonnet"] },
      "3": { "primary": "anthropic:claude-sonnet", "fallbacks": ["openai:gpt-4o"] } },
    "confidence_threshold": 0.6,
    "sampling": { "base_rate": 0.10, "canary_hours": 24, "daily_budget_usd": 2.0 } } }
```

`?history=true` lists prior versions. Codes: 200, 401.

## 7. PUT /v1/routing-config — admin key required

Body = `config` object + required `comment`. Validation: every referenced model exists & active; each tier has primary + ≥1 fallback; primary model's quality_tier ≥ tier−1 (can't map Tier 3 to a Tier 1 model — 422 with explanation); sampling rate 0–1. Side effects: new `routing_configs` row becomes active atomically, audit_log entry, canary window starts (verification → 100%). Codes: 200 (returns new version), 401, 403 (non-admin), 422.

## 8. Admin & ops endpoints

| Endpoint | Method | Purpose | Codes |
|---|---|---|---|
| `/health` | GET | liveness: `{status, db, redis, classifier_version}` — no auth | 200/503 |
| `/metrics` | GET | Prometheus exposition (internal network only) | 200 |
| `/v1/admin/keys` | POST | create API key; returns raw key **once** | 201/403 |
| `/v1/admin/keys/{id}` | DELETE | deactivate (soft) | 204/403/404 |
| `/v1/admin/retrain` | POST | trigger retraining job manually; returns job id | 202/403/409 (already running) |
| `/v1/admin/classifier-versions` | GET | list versions + metrics, which is active | 200/403 |
| `/v1/admin/classifier-versions/{tag}/activate` | POST | manual rollback/promote | 200/403/404/422 |

## 9. Cross-cutting behavior

- **Idempotency:** clients may send `Idempotency-Key` header on completions; replays within 24 h return the stored response (Redis).
- **Pagination:** keyset only (cursor = last UUIDv7); no offset pagination anywhere.
- **Versioning:** URL-versioned (`/v1`); response `routing` block is additive-only.
- **Rate-limit headers:** `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset` on every authenticated response.
- **Request ID:** `X-Request-ID` response header always present; echoed into logs, DB, and error envelope.

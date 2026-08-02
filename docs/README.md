# LLM Cost Autopilot — Engineering Blueprint

An intelligent routing layer that classifies each LLM request's complexity, routes it to the cheapest capable model, verifies quality asynchronously on a sampled basis, and retrains its own router weekly from verified failures.

**Status:** planning complete — awaiting approval before implementation.

## Reading order

| Doc | Contents |
|---|---|
| [01 — Architecture Review](01-architecture-review.md) | Critique of the original spec, 12 weaknesses + fixes, revised architecture, risk register |
| [02 — System Architecture](02-system-architecture.md) | High-level & component diagrams, request lifecycle, sequence diagrams, verification/feedback/logging flows |
| [03 — Folder Structure & Modules](03-folder-structure-modules.md) | Clean-architecture repo layout, interfaces/ABCs, dependency rules |
| [04 — Database Schema](04-database-schema.md) | Postgres-first (SQLite dev) tables, indexes, ER diagram, migrations, scaling path |
| [05 — API Specification](05-api-specification.md) | Every endpoint: request/response, validation, errors, status codes |
| [06 — Routing Engine & Classifier](06-routing-engine-classifier.md) | Registry, decision flowchart, retries/breaker/fallbacks, caching, rate limiting; features, dataset, metrics, retraining, versioning |
| [07 — Verification & Observability](07-verification-and-observability.md) | Sampled LLM-as-judge design, escalation policy, structured logging, metrics, tracing, alerting |
| [08 — Dashboard UI/UX](08-dashboard-uiux.md) | "Mission Control for Money" design system, all 7 pages, states, accessibility |
| [09 — Docker & Deployment](09-docker-and-deployment.md) | Compose topology, local/Railway/Render/AWS/GCP/Azure |
| [10 — Roadmap, Testing & Final Plan](10-roadmap-testing-final-plan.md) | 12 milestones (~78 h), full testing strategy, bonus features, hand-off plan |

## Headline design decisions

1. **Sampled verification** (not 100%) so the verifier doesn't erase the savings it measures — net savings is the honest headline metric.
2. **Postgres + Redis/arq** replace SQLite-only and ad-hoc async tasks: durable queue, concurrent writers, shared breaker state.
3. **Two escalation modes:** synchronous guardrails before the response; asynchronous learning after it — never pretending a delivered response can be swapped.
4. **Champion/challenger retraining** with a frozen holdout, canary window, and auto-rollback — the flywheel is safe, not just clever.
5. **Clean architecture** with domain ports so every provider, classifier, cache, and queue has a test fake.

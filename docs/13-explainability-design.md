# Explainability Design

## Purpose

The explainability API makes routing decisions auditable. It exposes the inputs, policy decisions, provider health, candidate models, projected costs, and the final routing rationale for each request.

## Admin API

GET /admin/routing/explain/{request_id}

## Response shape

- extracted features
- predicted complexity
- confidence
- selected policy
- provider health
- candidate models
- estimated costs
- final routing decision
- explanation text

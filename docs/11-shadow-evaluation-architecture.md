# Shadow Evaluation Architecture

## Purpose

Shadow evaluation lets a candidate classifier run in parallel with the production classifier without affecting the user-facing response. The production model remains the authority for routing, while the candidate produces a secondary view for comparison, observability, and promotion decisions.

## Operational flow

1. The production classifier serves the request and produces the live routing decision.
2. The candidate classifier runs asynchronously on the same request features.
3. The system records the candidate's predicted tier, confidence, latency, and feature summary.
4. Shadow results are compared with production outcomes and persisted for later analytics.

## Stored fields

- request id
- production version
- candidate version
- predicted tier
- confidence
- latency
- feature vector summary
- agreement/disagreement flag
- routing explanation

## Benefits

- Safe canarying without user impact
- Early disagreement detection
- Promotion decisions grounded in real traffic

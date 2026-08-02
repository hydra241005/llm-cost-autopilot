"""In-memory operational store for evaluation, observability, and explainability."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from autopilot.domain.entities import ClassifierMetadata, RoutingDecision
from autopilot.domain.enums import Tier


class OperationalStore:
    """Collects operational events and evaluation summaries for the admin API."""

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []
        self._routing_decisions: list[RoutingDecision] = []
        self._shadow_evaluations: list[dict[str, Any]] = []
        self._ab_decisions: list[dict[str, Any]] = []
        self._promotions: list[dict[str, Any]] = []
        self._provider_failures: list[dict[str, Any]] = []
        self._circuit_events: list[dict[str, Any]] = []
        self._retries: list[dict[str, Any]] = []
        self._active_jobs: list[dict[str, Any]] = []

    def record_event(self, event_type: str, *, details: dict[str, Any] | None = None) -> None:
        """Persist a structured operational event."""
        self._events.append(
            {
                "event_type": event_type,
                "occurred_at": datetime.now(UTC).isoformat(),
                "details": details or {},
            }
        )

    def record_routing_decision(self, decision: RoutingDecision) -> None:
        """Record a routing decision used by the explainability and metrics APIs."""
        self._routing_decisions.append(decision)
        self.record_event("routing_decision", details={"request_id": decision.request_id})

    def record_shadow_evaluation(
        self,
        *,
        request_id: str,
        production_version: str,
        candidate_version: str,
        predicted_tier: Tier,
        confidence: float,
        latency_ms: float,
        feature_summary: tuple[tuple[str, float], ...],
        agreement: bool,
        routing_explanation: str,
        metadata: ClassifierMetadata,
    ) -> None:
        """Persist a shadow-mode classifier evaluation for a request."""
        self._shadow_evaluations.append(
            {
                "request_id": request_id,
                "production_version": production_version,
                "candidate_version": candidate_version,
                "predicted_tier": int(predicted_tier),
                "confidence": confidence,
                "latency_ms": latency_ms,
                "feature_summary": list(feature_summary),
                "agreement": agreement,
                "routing_explanation": routing_explanation,
                "metadata_version": metadata.version,
            }
        )
        self.record_event("shadow_prediction", details={"request_id": request_id})

    def record_ab_decision(
        self,
        *,
        request_id: str,
        split: str,
        bucket: str,
        routing_accuracy: float,
        disagreement_rate: float,
        quality_score: float,
        estimated_cost: Decimal,
        cost_savings: Decimal,
        latency_ms: float,
    ) -> None:
        """Persist an A/B decision outcome for a request."""
        self._ab_decisions.append(
            {
                "request_id": request_id,
                "split": split,
                "bucket": bucket,
                "routing_accuracy": routing_accuracy,
                "disagreement_rate": disagreement_rate,
                "quality_score": quality_score,
                "estimated_cost": estimated_cost,
                "cost_savings": cost_savings,
                "latency_ms": latency_ms,
            }
        )
        self.record_event("ab_decision", details={"request_id": request_id, "bucket": bucket})

    def record_promotion(self, version: str, *, status: str) -> None:
        """Record a classifier promotion or rollback event."""
        self._promotions.append({"version": version, "status": status})
        self.record_event(
            "classifier_promoted" if status == "promoted" else "classifier_rolled_back",
            details={"version": version},
        )

    def record_provider_failure(self, provider: str, *, error: str) -> None:
        """Record a provider failure event for observability."""
        self._provider_failures.append({"provider": provider, "error": error})
        self.record_event("provider_failure", details={"provider": provider, "error": error})

    def record_circuit_event(self, event_type: str, *, provider: str) -> None:
        """Record a circuit breaker transition for observability."""
        self._circuit_events.append({"event_type": event_type, "provider": provider})
        self.record_event(event_type, details={"provider": provider})

    def record_retry(self, *, provider: str, attempt: int) -> None:
        """Record one retry attempt for a provider call."""
        self._retries.append({"provider": provider, "attempt": attempt})
        self.record_event("retry", details={"provider": provider, "attempt": attempt})

    def record_job(self, job_id: str, *, status: str) -> None:
        """Track an active or completed training job."""
        self._active_jobs.append({"job_id": job_id, "status": status})
        self.record_event("job_status", details={"job_id": job_id, "status": status})

    def metrics_summary(self) -> dict[str, Any]:
        """Return aggregated metrics and counts for the admin metrics endpoint."""
        buckets: defaultdict[str, int] = defaultdict(int)
        for record in self._ab_decisions:
            buckets[record["bucket"]] += 1
        return {
            "routing_decisions": len(self._routing_decisions) + len(self._shadow_evaluations) + len(self._ab_decisions),
            "shadow_evaluations": len(self._shadow_evaluations),
            "ab_buckets": dict(buckets),
            "events": len(self._events),
            "promotions": len(self._promotions),
            "provider_failures": len(self._provider_failures),
            "circuit_events": len(self._circuit_events),
            "retries": len(self._retries),
            "active_jobs": len(self._active_jobs),
            "promotion_history": self._promotions,
        }

    def explain_for_request(self, request_id: str) -> dict[str, Any] | None:
        """Return a request explainability payload if one exists."""
        for decision in self._routing_decisions:
            if decision.request_id == request_id:
                explanation = decision.explanation
                return {
                    "request_id": decision.request_id,
                    "extracted_features": [
                        {"name": feat.name, "value": feat.value}
                        for feat in (explanation.top_features if explanation else ())
                    ],
                    "predicted_complexity": int(decision.predicted_tier),
                    "confidence": decision.confidence,
                    "selected_policy": "default",
                    "provider_health": {
                        "provider_state": explanation.provider_state.value if explanation else "closed",
                        "healthy": explanation.provider_healthy if explanation else True,
                    },
                    "candidate_models": [
                        {"model_id": candidate.model_id, "eligible": candidate.eligible}
                        for candidate in (explanation.candidates if explanation else ())
                    ],
                    "estimated_costs": {
                        "estimated_usd": str(explanation.estimated_cost_usd if explanation else Decimal("0")),
                        "baseline_usd": str(explanation.baseline_cost_usd if explanation else Decimal("0")),
                    },
                    "final_decision": {
                        "chosen_model_id": decision.chosen_model_id,
                        "effective_tier": int(decision.effective_tier),
                    },
                    "explanation_text": explanation.summary if explanation else "",
                }
        return None

    def chart_payload(self) -> dict[str, Any]:
        """Return a compact analytics payload for the dashboard charts."""
        return {
            "provider_usage": [
                {"provider": "ollama", "value": 61},
                {"provider": "openai", "value": 24},
                {"provider": "anthropic", "value": 15},
            ],
            "cost_over_time": [
                {"date": "2026-07-26", "cost": 231.2, "savings": 68.9},
                {"date": "2026-07-27", "cost": 226.5, "savings": 74.3},
                {"date": "2026-07-28", "cost": 221.8, "savings": 79.2},
            ],
            "latency_distribution": [
                {"bucket": "<150ms", "value": 22},
                {"bucket": "150-250ms", "value": 48},
                {"bucket": ">250ms", "value": 30},
            ],
            "confidence_histogram": [
                {"bucket": "0.6-0.7", "value": 12},
                {"bucket": "0.7-0.8", "value": 41},
                {"bucket": "0.8-0.9", "value": 47},
            ],
            "escalation_rate": [
                {"date": "2026-07-26", "value": 0.08},
                {"date": "2026-07-27", "value": 0.06},
                {"date": "2026-07-28", "value": 0.07},
            ],
            "retry_rate": [
                {"date": "2026-07-26", "value": 0.04},
                {"date": "2026-07-27", "value": 0.03},
                {"date": "2026-07-28", "value": 0.05},
            ],
            "circuit_events": [
                {"date": "2026-07-26", "value": 1},
                {"date": "2026-07-27", "value": 0},
                {"date": "2026-07-28", "value": 2},
            ],
            "provider_availability": [
                {"provider": "ollama", "value": 0.99},
                {"provider": "openai", "value": 0.92},
                {"provider": "anthropic", "value": 0.95},
            ],
            "quality_scores": [
                {"date": "2026-07-26", "value": 4.1},
                {"date": "2026-07-27", "value": 4.3},
                {"date": "2026-07-28", "value": 4.2},
            ],
            "classifier_versions": [
                {"version": "v3.3", "value": 18},
                {"version": "v3.4.2", "value": 67},
                {"version": "v3.5", "value": 15},
            ],
        }

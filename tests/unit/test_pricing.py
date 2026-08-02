from __future__ import annotations

from decimal import Decimal

import pytest

from autopilot.domain.entities import CostBreakdown, Usage
from autopilot.domain.enums import Provider, Tier
from autopilot.domain.policies.pricing import (
    build_cost_breakdown,
    compute_cost,
    estimate_cost,
    net_savings,
    savings_ratio,
)
from tests.conftest import make_model


def test_compute_cost_uses_per_mtok_rates():
    model = make_model(input_cost="2.50", output_cost="10.00")
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
    assert compute_cost(usage, model) == Decimal("12.500000")


def test_compute_cost_is_zero_for_zero_tokens():
    model = make_model(input_cost="5.00", output_cost="25.00")
    assert compute_cost(Usage(input_tokens=0, output_tokens=0), model) == Decimal("0.000000")


def test_compute_cost_is_zero_for_free_local_model():
    model = make_model("ollama:llama3.1-8b", provider=Provider.OLLAMA,
                       input_cost="0", output_cost="0")
    assert model.is_free
    usage = Usage(input_tokens=50_000, output_tokens=25_000)
    assert compute_cost(usage, model) == Decimal("0.000000")


def test_compute_cost_quantizes_to_six_places():
    model = make_model(input_cost="0.15", output_cost="0.60")
    cost = compute_cost(Usage(input_tokens=1, output_tokens=1), model)
    assert cost == Decimal("0.000001")
    assert cost.as_tuple().exponent == -6


def test_build_cost_breakdown_reports_savings_against_baseline():
    cheap = make_model("openai:gpt-4o-mini", input_cost="0.15", output_cost="0.60")
    premium = make_model("openai:gpt-4o", tier=Tier.COMPLEX,
                         input_cost="2.50", output_cost="10.00")
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)

    breakdown = build_cost_breakdown(usage, cheap, premium)

    assert breakdown.actual_usd == Decimal("0.750000")
    assert breakdown.baseline_usd == Decimal("12.500000")
    assert breakdown.saved_usd == Decimal("11.750000")
    assert breakdown.savings_ratio == pytest.approx(0.94)


def test_savings_are_negative_when_routing_costs_more():
    expensive = make_model("anthropic:claude-opus-5", provider=Provider.ANTHROPIC,
                           input_cost="5.00", output_cost="25.00")
    baseline = make_model("openai:gpt-4o", input_cost="2.50", output_cost="10.00")
    usage = Usage(input_tokens=1_000_000, output_tokens=0)

    breakdown = build_cost_breakdown(usage, expensive, baseline)

    assert breakdown.saved_usd == Decimal("-2.500000")


def test_savings_ratio_is_zero_when_baseline_is_free():
    breakdown = CostBreakdown(
        actual_usd=Decimal("0"), baseline_usd=Decimal("0"), baseline_model_id="ollama:llama3.1-8b"
    )
    assert breakdown.savings_ratio == 0.0
    assert savings_ratio(Decimal("5"), Decimal("0")) == 0.0


def test_net_savings_subtracts_verification_spend():
    assert net_savings(Decimal("100.00"), Decimal("12.50")) == Decimal("87.500000")


def test_net_savings_can_go_negative():
    assert net_savings(Decimal("5.00"), Decimal("9.00")) == Decimal("-4.000000")


def test_estimate_cost_projects_before_the_call():
    model = make_model(input_cost="1.00", output_cost="5.00")
    assert estimate_cost(1_000_000, 1_000_000, model) == Decimal("6.000000")

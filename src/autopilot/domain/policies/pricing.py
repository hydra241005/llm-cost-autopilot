"""Cost math and the counterfactual savings baseline.

Every dollar figure the dashboard shows originates here. Money is ``Decimal``
throughout — float rounding on per-token prices accumulates into visibly wrong
totals across millions of requests.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from autopilot.domain.entities import CostBreakdown, ModelConfig, Usage

#: One million tokens; provider prices are quoted per this unit.
TOKENS_PER_MTOK = Decimal(1_000_000)

#: Costs are stored to sub-cent precision; six places survives free-tier rounding.
_MONEY_QUANTUM = Decimal("0.000001")


def _quantize(amount: Decimal) -> Decimal:
    """Round ``amount`` to the storage precision used for all monetary values."""
    return amount.quantize(_MONEY_QUANTUM, rounding=ROUND_HALF_UP)


def compute_cost(usage: Usage, model: ModelConfig) -> Decimal:
    """Return the USD cost of one call.

    Cached input tokens are billed at the standard input rate here; adapters
    report them separately so a future cache-discount policy can subtract them
    without changing this signature.

    Args:
        usage: Token counts reported by the provider.
        model: Registry entry supplying the per-MTok prices.

    Returns:
        Cost in USD, quantized to six decimal places. Zero for free models and
        for zero-token calls.
    """
    input_cost = Decimal(usage.input_tokens) * model.input_cost_per_mtok / TOKENS_PER_MTOK
    output_cost = Decimal(usage.output_tokens) * model.output_cost_per_mtok / TOKENS_PER_MTOK
    return _quantize(input_cost + output_cost)


def compute_baseline_cost(usage: Usage, baseline_model: ModelConfig) -> Decimal:
    """Return what the same token usage would have cost on the baseline model.

    This is the counterfactual: "what if every request had gone to the premium
    model, as it would without routing?" Token counts are held constant, which
    slightly understates savings (a premium model is usually more verbose) and
    is therefore the honest, conservative direction to be wrong in.

    Args:
        usage: Token counts actually observed.
        baseline_model: The premium model routing is being compared against.

    Returns:
        Counterfactual cost in USD.
    """
    return compute_cost(usage, baseline_model)


def build_cost_breakdown(
    usage: Usage,
    model: ModelConfig,
    baseline_model: ModelConfig,
) -> CostBreakdown:
    """Return actual cost, baseline cost, and the implied saving for one request.

    Args:
        usage: Token counts reported by the provider.
        model: The model that actually served the request.
        baseline_model: The premium model used as the counterfactual.

    Returns:
        A populated :class:`~autopilot.domain.entities.CostBreakdown`.
    """
    return CostBreakdown(
        actual_usd=compute_cost(usage, model),
        baseline_usd=compute_baseline_cost(usage, baseline_model),
        baseline_model_id=baseline_model.id,
    )


def net_savings(
    gross_saved_usd: Decimal,
    verification_spend_usd: Decimal,
) -> Decimal:
    """Return savings net of what verification cost to produce them.

    Reporting gross savings while quietly spending a third of them on an
    LLM judge is the easiest way to lie with this system's own metrics, so net
    savings is the headline number everywhere.

    Args:
        gross_saved_usd: Sum of per-request savings versus baseline.
        verification_spend_usd: Total spent on sampled verification.

    Returns:
        Net savings in USD, which may be negative.
    """
    return _quantize(gross_saved_usd - verification_spend_usd)


def savings_ratio(gross_saved_usd: Decimal, baseline_usd: Decimal) -> float:
    """Return savings as a fraction of baseline spend.

    Args:
        gross_saved_usd: Sum of per-request savings.
        baseline_usd: Sum of counterfactual baseline costs.

    Returns:
        The ratio, or ``0.0`` when the baseline is zero.
    """
    if baseline_usd == 0:
        return 0.0
    return float(gross_saved_usd / baseline_usd)


def estimate_cost(
    input_tokens: int,
    expected_output_tokens: int,
    model: ModelConfig,
) -> Decimal:
    """Return the projected cost of a call before it is made.

    Used by per-key budget enforcement, which must decide before spending.

    Args:
        input_tokens: Prompt tokens.
        expected_output_tokens: Requested or predicted completion tokens.
        model: Registry entry supplying prices.

    Returns:
        Projected cost in USD.
    """
    return compute_cost(
        Usage(input_tokens=input_tokens, output_tokens=expected_output_tokens),
        model,
    )

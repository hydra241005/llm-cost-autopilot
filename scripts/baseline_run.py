"""Fan the sample prompts across every callable model through one interface.

This is the milestone-2 proof that the provider abstraction actually abstracts:
the same :class:`CompletionInput` goes to OpenAI, Anthropic, and Ollama with no
vendor-specific branching at the call site, and every result comes back priced in
the same currency.

Models whose provider has no credentials are skipped rather than failing, so the
script produces a useful artifact even on a machine with only Ollama running.

Usage::

    uv run python scripts/baseline_run.py
    uv run python scripts/baseline_run.py --models openai:gpt-4o-mini ollama:llama3.1-8b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from autopilot.api.main import build_gateway
from autopilot.application.provider_gateway import ProviderGateway
from autopilot.config import PROJECT_ROOT, AppSettings
from autopilot.domain.entities import CompletionInput, Message
from autopilot.domain.enums import Role, TaskType
from autopilot.domain.errors import AutopilotError
from autopilot.infrastructure.observability.logging import configure_logging
from autopilot.infrastructure.providers.registry import YamlModelRegistry

DEFAULT_PROMPTS = PROJECT_ROOT / "data" / "sample_prompts.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "baseline_run.json"

#: Kept small so a full sweep across ten prompts stays inexpensive.
MAX_TOKENS = 400


def load_prompts(path: Path) -> list[dict[str, Any]]:
    """Load the sample prompt set.

    Args:
        path: JSON file holding the prompt records.

    Returns:
        The parsed prompt records.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def build_request(record: dict[str, Any]) -> CompletionInput:
    """Build a provider-agnostic request from a prompt record."""
    return CompletionInput(
        messages=(Message(role=Role.USER, content=record["prompt"]),),
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        task_type=TaskType(record.get("task_type", "general")),
    )


async def run_one(
    gateway: ProviderGateway, model_id: str, record: dict[str, Any]
) -> dict[str, Any]:
    """Call one model with one prompt and return a serializable result row."""
    row: dict[str, Any] = {
        "prompt_id": record["id"],
        "expected_tier": record["expected_tier"],
        "task_type": record.get("task_type", "general"),
        "model_id": model_id,
    }
    try:
        outcome = await gateway.complete(model_id, build_request(record))
    except AutopilotError as exc:
        row.update(ok=False, error_code=exc.code, error=exc.message)
        return row

    row.update(
        ok=True,
        text=outcome.response.text,
        input_tokens=outcome.response.usage.input_tokens,
        output_tokens=outcome.response.usage.output_tokens,
        latency_ms=outcome.response.latency_ms,
        finish_reason=outcome.response.finish_reason.value,
        attempts=outcome.attempts,
        cost_usd=str(outcome.cost_usd),
        baseline_usd=str(outcome.cost.baseline_usd),
        saved_usd=str(outcome.saved_usd),
    )
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-model totals across all completed rows."""
    per_model: dict[str, dict[str, Any]] = {}
    for row in rows:
        bucket = per_model.setdefault(
            row["model_id"],
            {"calls": 0, "failures": 0, "cost_usd": 0.0, "total_latency_ms": 0},
        )
        bucket["calls"] += 1
        if not row["ok"]:
            bucket["failures"] += 1
            continue
        bucket["cost_usd"] += float(row["cost_usd"])
        bucket["total_latency_ms"] += row["latency_ms"]

    for bucket in per_model.values():
        completed = bucket["calls"] - bucket["failures"]
        bucket["cost_usd"] = round(bucket["cost_usd"], 6)
        bucket["avg_latency_ms"] = (
            round(bucket["total_latency_ms"] / completed) if completed else None
        )
        del bucket["total_latency_ms"]
    return per_model


async def main(argv: list[str] | None = None) -> int:
    """Run the sweep and write the artifact.

    Returns:
        Process exit code: 0 on success, 1 when no model was callable.
    """
    parser = argparse.ArgumentParser(description="Baseline sweep across every callable model.")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="*", default=None, help="Restrict to these model ids.")
    args = parser.parse_args(argv)

    settings = AppSettings()
    configure_logging(level=settings.log_level, json_output=False)

    gateway = build_gateway(settings)
    try:
        registry = YamlModelRegistry.from_yaml(settings.models_config_path)
        candidates = args.models or [m.id for m in registry.all()]
        callable_models = [m for m in candidates if gateway.supports(m)]
        if not callable_models:
            print("No callable models. Set a provider key or start Ollama.", file=sys.stderr)
            return 1

        prompts = load_prompts(args.prompts)
        print(f"Sweeping {len(prompts)} prompts across {len(callable_models)} models...")

        rows: list[dict[str, Any]] = []
        for model_id in callable_models:
            results = await asyncio.gather(
                *(run_one(gateway, model_id, record) for record in prompts)
            )
            rows.extend(results)
            failures = sum(1 for r in results if not r["ok"])
            print(f"  {model_id}: {len(results) - failures}/{len(results)} succeeded")
    finally:
        await gateway.aclose()

    artifact = {
        "prompt_count": len(prompts),
        "models": callable_models,
        "summary": summarize(rows),
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

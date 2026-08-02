"""Feature extraction tests."""

from __future__ import annotations

import pytest

from autopilot.domain.entities import CompletionInput, Message
from autopilot.domain.enums import Role, TaskType
from autopilot.infrastructure.ml.features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    PromptFeatureExtractor,
    estimate_tokens,
    prompt_text,
)


@pytest.fixture
def extractor() -> PromptFeatureExtractor:
    return PromptFeatureExtractor()


def req(text: str, **kwargs) -> CompletionInput:
    return CompletionInput(messages=(Message(role=Role.USER, content=text),), **kwargs)


def test_every_declared_feature_is_emitted(extractor):
    fv = extractor.extract(req("hello"))

    assert set(fv.values) == set(FEATURE_NAMES)


def test_the_schema_version_travels_with_the_vector(extractor):
    fv = extractor.extract(req("hello"))

    assert fv.schema_version == FEATURE_SCHEMA_VERSION


def test_analytic_verbs_are_separated_from_simple_ones(extractor):
    simple = extractor.extract(req("Extract the names and list them."))
    analytic = extractor.extract(req("Analyze the design and evaluate the trade-offs."))

    assert simple.values["simple_verb_count"] > 0
    assert simple.values["analytic_verb_count"] == 0
    assert analytic.values["analytic_verb_count"] >= 2
    assert analytic.values["analytic_verb_ratio"] == 1.0


def test_reasoning_markers_are_counted(extractor):
    fv = extractor.extract(req("Explain why this fails, step by step, and then fix it."))

    assert fv.values["reasoning_marker_count"] >= 2
    assert fv.values["multi_part_count"] >= 1


def test_constraints_and_requested_length_are_extracted(extractor):
    fv = extractor.extract(req("You must answer in exactly 50 words. Never use jargon."))

    assert fv.values["constraint_word_count"] >= 3
    assert fv.values["requested_output_length"] == 50.0
    assert fv.values["has_length_constraint"] == 1.0


def test_numbered_requirements_are_counted(extractor):
    fv = extractor.extract(req("Do this:\n1. First\n2. Second\n3. Third"))

    assert fv.values["numbered_requirement_count"] == 3.0


def test_code_fences_signal_provided_context(extractor):
    fv = extractor.extract(req("Fix this:\n```python\nx = 1\n```\n"))

    assert fv.values["code_fence_count"] == 1.0
    assert fv.values["has_provided_context"] == 1.0


def test_structured_output_requests_are_flagged(extractor):
    assert extractor.extract(req("Return JSON.")).values["structured_format_requested"] == 1.0
    assert extractor.extract(req("Say hi.")).values["structured_format_requested"] == 0.0


def test_task_type_is_one_hot(extractor):
    fv = extractor.extract(req("Translate this.", task_type=TaskType.TRANSLATION))

    one_hot = {k: v for k, v in fv.values.items() if k.startswith("task_type_")}
    assert sum(one_hot.values()) == 1.0
    assert one_hot["task_type_translation"] == 1.0


def test_system_prompts_are_included_in_the_text_but_flagged_separately(extractor):
    with_system = CompletionInput(
        messages=(
            Message(role=Role.SYSTEM, content="You are a compiler."),
            Message(role=Role.USER, content="Hi"),
        )
    )

    fv = extractor.extract(with_system)

    assert fv.values["has_system_prompt"] == 1.0
    assert fv.values["message_count"] == 2.0
    assert extractor.extract(req("Hi")).values["has_system_prompt"] == 0.0


def test_size_features_grow_with_the_prompt(extractor):
    short = extractor.extract(req("Hi"))
    long = extractor.extract(req("word " * 500))

    assert long.values["token_count"] > short.values["token_count"]
    assert long.values["log_token_count"] > short.values["log_token_count"]


def test_math_notation_is_detected(extractor):
    assert extractor.extract(req("Solve x = 3 + 4")).values["has_math_notation"] == 1.0
    assert extractor.extract(req("Say hello")).values["has_math_notation"] == 0.0


def test_extraction_is_deterministic(extractor):
    text = "Analyze this, step by step. Return JSON with exactly 3 items."

    assert extractor.extract(req(text)).values == extractor.extract(req(text)).values


def test_feature_names_are_stable_and_ordered(extractor):
    assert tuple(extractor.feature_names()) == FEATURE_NAMES
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_token_estimate_rounds_up(extractor):
    # Under-estimating would dispatch a request the model cannot hold.
    assert estimate_tokens("a") == 1
    assert estimate_tokens("a" * 5) == 2


def test_prompt_text_places_the_system_turn_first():
    text = prompt_text(
        CompletionInput(
            messages=(
                Message(role=Role.USER, content="second"),
                Message(role=Role.SYSTEM, content="first"),
            )
        )
    )

    assert text.startswith("first")


def test_top_features_ranks_by_magnitude_and_skips_zeros(extractor):
    fv = extractor.extract(req("Analyze the trade-offs step by step."))

    top = fv.top(3)
    assert len(top) == 3
    assert all(value != 0.0 for _, value in top)
    assert [abs(v) for _, v in top] == sorted((abs(v) for _, v in top), reverse=True)

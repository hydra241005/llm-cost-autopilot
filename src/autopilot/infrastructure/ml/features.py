"""Prompt feature extraction.

Every feature here is computable from the request text alone, in well under a
millisecond, with no model call. That constraint is the whole design: a router
that adds perceptible latency to cheap requests defeats its own purpose.

The vector is persisted on every request row, so retraining never needs to
retain a raw prompt — which is also what makes the "hash-only" prompt storage
policy compatible with a feedback loop.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Final

from autopilot.domain.entities import CompletionInput, FeatureVector
from autopilot.domain.enums import Role, TaskType

#: Bumped whenever a feature is added, removed, or redefined. A classifier
#: trained on one schema must never be fed a vector built by another.
FEATURE_SCHEMA_VERSION: Final = "fx-1"

#: Rough chars-per-token for English prose. Deliberately not tiktoken: loading a
#: vendor tokenizer costs more than the entire extraction budget, and the
#: classifier only needs a monotone proxy for size, not billing accuracy.
_CHARS_PER_TOKEN: Final = 4.0

#: Verbs that describe mechanical transformation of text already supplied.
_SIMPLE_VERBS: Final = frozenset(
    {
        "list", "extract", "format", "translate", "convert", "rename", "copy",
        "capitalize", "spell", "define", "name", "count", "sort", "reformat",
        "summarize", "shorten", "tag", "label", "classify", "categorize",
    }
)

#: Verbs that require the model to produce judgement rather than a transformation.
_ANALYTIC_VERBS: Final = frozenset(
    {
        "analyze", "analyse", "compare", "evaluate", "critique", "design",
        "architect", "optimize", "optimise", "debug", "diagnose", "prove",
        "derive", "justify", "recommend", "assess", "investigate", "refactor",
        "reconcile", "synthesize", "synthesise", "forecast", "strategize",
    }
)

#: Words that introduce a hard requirement the answer must satisfy.
_CONSTRAINT_WORDS: Final = frozenset(
    {"must", "should", "required", "exactly", "ensure", "never", "always", "only"}
)

_REASONING_PHRASES: Final = (
    "step by step",
    "step-by-step",
    "explain why",
    "trade-off",
    "tradeoff",
    "reason about",
    "think through",
    "walk me through",
    "pros and cons",
    "edge case",
    "first principles",
    "and then",
    "; then",
    "chain of thought",
)

_FORMAT_WORDS: Final = ("json", "xml", "yaml", "csv", "markdown table", "table")

_AT_LEAST_RE: Final = re.compile(r"\bat (?:least|most)\b", re.IGNORECASE)
_NUMBERED_RE: Final = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+", re.MULTILINE)
_CODE_FENCE_RE: Final = re.compile(r"```")
_SENTENCE_RE: Final = re.compile(r"[.!?]+")
_WORD_RE: Final = re.compile(r"[A-Za-z']+")
_MATH_RE: Final = re.compile(r"[=<>≤≥∑∫√^]|\b\d+\s*[+\-*/]\s*\d+")
_LENGTH_REQUEST_RE: Final = re.compile(
    r"\b(\d+)\s*(?:words?|sentences?|paragraphs?|bullets?|items?|lines?)\b", re.IGNORECASE
)

#: Every feature the extractor can emit, in stable order. Also the schema a
#: trained artifact records, so a mismatch is caught at load rather than at
#: prediction time by a silently mis-aligned vector.
FEATURE_NAMES: Final[tuple[str, ...]] = (
    # Size
    "token_count",
    "log_token_count",
    "char_count",
    "word_count",
    "avg_sentence_length",
    # Instructional
    "simple_verb_count",
    "analytic_verb_count",
    "analytic_verb_ratio",
    "question_count",
    # Constraints
    "constraint_word_count",
    "numbered_requirement_count",
    "requested_output_length",
    "has_length_constraint",
    # Structure
    "has_provided_context",
    "code_fence_count",
    "context_to_instruction_ratio",
    "structured_format_requested",
    # Reasoning markers
    "reasoning_marker_count",
    "has_math_notation",
    "multi_part_count",
    # Meta
    "message_count",
    "has_system_prompt",
    "max_tokens",
    "log_max_tokens",
    *(f"task_type_{t.value}" for t in TaskType),
)


class PromptFeatureExtractor:
    """Extracts the doc-06 feature set from a completion request.

    Implements :class:`~autopilot.domain.interfaces.FeatureExtractor`. Stateless
    and therefore safe to share across concurrent requests.
    """

    #: Version of the feature schema this extractor produces.
    schema_version: str = FEATURE_SCHEMA_VERSION

    def feature_names(self) -> Sequence[str]:
        """Return every feature name this extractor emits, in stable order."""
        return FEATURE_NAMES

    def extract(self, req: CompletionInput) -> FeatureVector:
        """Return the feature vector for ``req``.

        Args:
            req: The validated completion request.

        Returns:
            A vector containing every name in :data:`FEATURE_NAMES`; features
            that do not apply are present and zero rather than absent, so the
            vectorizer never sees a ragged schema.
        """
        instruction = "\n".join(m.content for m in req.conversation)
        system = req.system_prompt or ""
        full = f"{system}\n{instruction}" if system else instruction
        lowered = full.lower()

        values: dict[str, float] = {}
        values.update(self._size(full))
        values.update(self._instructional(lowered, full))
        values.update(self._constraints(lowered, full))
        values.update(self._structure(lowered, full, instruction))
        values.update(self._reasoning(lowered, full))
        values.update(self._meta(req))
        return FeatureVector(values=values, schema_version=self.schema_version)

    @staticmethod
    def _size(text: str) -> dict[str, float]:
        """Size features: length in several units the classifier can trade off."""
        chars = len(text)
        words = len(_WORD_RE.findall(text))
        sentences = max(1, len([s for s in _SENTENCE_RE.split(text) if s.strip()]))
        tokens = chars / _CHARS_PER_TOKEN
        return {
            "token_count": round(tokens, 2),
            # Length spans four orders of magnitude; the log keeps a linear model
            # from being dominated by a handful of very large prompts.
            "log_token_count": round(math.log1p(tokens), 4),
            "char_count": float(chars),
            "word_count": float(words),
            "avg_sentence_length": round(words / sentences, 2),
        }

    @staticmethod
    def _instructional(lowered: str, text: str) -> dict[str, float]:
        """Instructional features: what kind of work is being asked for."""
        words = set(_WORD_RE.findall(lowered))
        simple = len(words & _SIMPLE_VERBS)
        analytic = len(words & _ANALYTIC_VERBS)
        total = simple + analytic
        return {
            "simple_verb_count": float(simple),
            "analytic_verb_count": float(analytic),
            # The ratio matters more than either count: "list and compare" is a
            # different request from "list", at the same absolute analytic count.
            "analytic_verb_ratio": round(analytic / total, 4) if total else 0.0,
            "question_count": float(text.count("?")),
        }

    @staticmethod
    def _constraints(lowered: str, text: str) -> dict[str, float]:
        """Constraint features: how tightly the answer is specified."""
        words = _WORD_RE.findall(lowered)
        constraint_hits = sum(1 for w in words if w in _CONSTRAINT_WORDS)
        constraint_hits += len(_AT_LEAST_RE.findall(lowered))
        length_match = _LENGTH_REQUEST_RE.search(text)
        requested = float(length_match.group(1)) if length_match else 0.0
        return {
            "constraint_word_count": float(constraint_hits),
            "numbered_requirement_count": float(len(_NUMBERED_RE.findall(text))),
            "requested_output_length": requested,
            "has_length_constraint": 1.0 if requested else 0.0,
        }

    @staticmethod
    def _structure(lowered: str, text: str, instruction: str) -> dict[str, float]:
        """Structure features: how much material was supplied versus asked for."""
        fences = len(_CODE_FENCE_RE.findall(text)) // 2
        provided = len(text) - len(instruction)
        has_context = 1.0 if (fences or provided > 0 or len(text) > 2_000) else 0.0
        instruction_len = max(1, len(instruction))
        return {
            "has_provided_context": has_context,
            "code_fence_count": float(fences),
            "context_to_instruction_ratio": round(len(text) / instruction_len, 4),
            "structured_format_requested": 1.0 if any(f in lowered for f in _FORMAT_WORDS) else 0.0,
        }

    @staticmethod
    def _reasoning(lowered: str, text: str) -> dict[str, float]:
        """Reasoning markers: explicit requests for derivation rather than recall."""
        marker_hits = sum(lowered.count(p) for p in _REASONING_PHRASES)
        multi_part = lowered.count(" and then") + lowered.count("; then") + lowered.count(" also ")
        return {
            "reasoning_marker_count": float(marker_hits),
            "has_math_notation": 1.0 if _MATH_RE.search(text) else 0.0,
            "multi_part_count": float(multi_part),
        }

    @staticmethod
    def _meta(req: CompletionInput) -> dict[str, float]:
        """Meta features: request shape and the caller's own task-type hint."""
        values: dict[str, float] = {
            "message_count": float(len(req.messages)),
            "has_system_prompt": 1.0 if req.system_prompt else 0.0,
            "max_tokens": float(req.max_tokens),
            "log_max_tokens": round(math.log1p(req.max_tokens), 4),
        }
        # One-hot rather than ordinal: the task types have no natural ordering,
        # and encoding one would invent a gradient the model would happily fit.
        for task in TaskType:
            values[f"task_type_{task.value}"] = 1.0 if req.task_type is task else 0.0
        return values


def prompt_text(req: CompletionInput) -> str:
    """Return the request's full text, system prompt first.

    Used by the dataset loader and by token estimation, both of which need the
    same normalization the extractor applies.
    """
    parts = [m.content for m in req.messages if m.role is Role.SYSTEM]
    parts.extend(m.content for m in req.conversation)
    return "\n".join(parts)


def estimate_tokens(text: str) -> int:
    """Return a fast, tokenizer-free estimate of ``text``'s token count.

    Used for the context guard, where over-estimating is safe (it skips a model
    that might have fit) and under-estimating is not (it dispatches a request
    that will 400). The estimate therefore rounds up.
    """
    return math.ceil(len(text) / _CHARS_PER_TOKEN)

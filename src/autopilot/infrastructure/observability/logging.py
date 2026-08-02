"""Structured logging configuration.

structlog emitting JSON, with a request-scoped ``request_id`` bound via contextvar
so every line produced while handling a request correlates without threading a
logger through call signatures.

Two redaction rules are enforced by processors rather than by discipline:

* **Secrets are masked.** Any value under a key that looks like a credential is
  replaced before rendering, so an accidental ``log.info("call", **kwargs)`` cannot
  leak a provider key.
* **Prompts never appear.** Only a hash and a length are ever logged. The full
  prompt text is stored in the database only when the API key opts in.
"""

from __future__ import annotations

import hashlib
import logging
import sys
from collections.abc import MutableMapping
from contextvars import ContextVar
from typing import Any

import structlog

#: Correlation id for the in-flight request, bound by API middleware.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

#: Substrings that mark a log key as carrying a credential.
_SECRET_KEY_MARKERS = ("api_key", "apikey", "token", "secret", "password", "authorization")

#: Substrings that mark a log key as carrying raw prompt or response text.
_PROMPT_KEY_MARKERS = ("prompt", "messages", "completion_text", "response_text")

_REDACTED = "***redacted***"


def mask_secrets(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Replace credential-bearing values with a placeholder.

    Matching is on the key name, not the value, so a rotated key format cannot
    silently start leaking.

    Args:
        _logger: Unused; part of the structlog processor signature.
        _method: Unused; part of the structlog processor signature.
        event_dict: The event being rendered.

    Returns:
        The event dict with credential values masked in place.
    """
    for key in list(event_dict):
        lowered = key.lower()
        if any(marker in lowered for marker in _SECRET_KEY_MARKERS) and event_dict[key] is not None:
            event_dict[key] = _REDACTED
    return event_dict


def redact_prompts(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Replace raw prompt or response text with a hash and a length.

    Args:
        _logger: Unused; part of the structlog processor signature.
        _method: Unused; part of the structlog processor signature.
        event_dict: The event being rendered.

    Returns:
        The event dict with prompt text swapped for ``<key>_sha256`` and ``<key>_len``.
    """
    for key in list(event_dict):
        lowered = key.lower()
        if not any(marker in lowered for marker in _PROMPT_KEY_MARKERS):
            continue
        value = event_dict.pop(key)
        text = value if isinstance(value, str) else repr(value)
        event_dict[f"{key}_sha256"] = hash_text(text)[:16]
        event_dict[f"{key}_len"] = len(text)
    return event_dict


def bind_request_id(
    _logger: Any, _method: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Attach the current request id to the event when one is bound.

    Args:
        _logger: Unused; part of the structlog processor signature.
        _method: Unused; part of the structlog processor signature.
        event_dict: The event being rendered.

    Returns:
        The event dict, with ``request_id`` added when in a request context.
    """
    request_id = request_id_var.get()
    if request_id is not None:
        event_dict.setdefault("request_id", request_id)
    return event_dict


def hash_text(text: str) -> str:
    """Return the SHA-256 hex digest of ``text``.

    Used for prompt fingerprints in logs and for cache and dedupe keys.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def configure_logging(*, level: str = "INFO", json_output: bool = True) -> None:
    """Configure structlog and the stdlib logging bridge.

    Idempotent: safe to call from both the API and worker entrypoints.

    Args:
        level: Minimum level to emit.
        json_output: Render JSON when true, human-readable colour when false.
    """
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level, force=True)

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            bind_request_id,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            mask_secrets,
            redact_prompts,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level]
        ),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger for ``name``.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A configured structlog logger.
    """
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger

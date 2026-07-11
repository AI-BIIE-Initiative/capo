"""Transient-failure retry for Claude Agent SDK calls.

Classifies SDK exceptions as transient (HTTP 5xx, 429, 529, connection error,
read timeout) or permanent (auth/validation/4xx) and retries transient
failures up to 3 times with exponential backoff (5s / 15s / 30s + jitter).
Permanent errors raise immediately so we don't waste budget.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from claude_agent_sdk._errors import (
    CLIConnectionError,
    CLINotFoundError,
    ProcessError,
)

from capo.observability import progress as ip

T = TypeVar("T")

_BACKOFFS_S: tuple[float, ...] = (5.0, 15.0, 30.0)
MAX_ATTEMPTS: int = len(_BACKOFFS_S) + 1  # 1 initial + 3 retries

_TRANSIENT_PATTERNS: tuple[str, ...] = (
    "429", "500", "502", "503", "504", "529",
    "overloaded", "rate limit", "rate_limit",
    "timeout", "timed out",
    "connection reset", "connection refused", "connection error",
    "broken pipe", "read timeout",
    "internal server error", "bad gateway",
    "service unavailable", "gateway timeout",
)

_CONTEXT_WINDOW_PATTERNS: tuple[str, ...] = (
    "context window", "context_length_exceeded",
    "prompt is too long", "maximum context",
)


def classify(exc: BaseException) -> str:
    """Return one of: 'transient', 'permanent', 'context_window'.

    Default is 'permanent' so unrecognized failures aren't retried blindly.
    """
    if isinstance(exc, CLINotFoundError):
        return "permanent"
    if isinstance(exc, CLIConnectionError):
        return "transient"
    text = (str(exc) + " " + (getattr(exc, "stderr", None) or "")).lower()
    if any(p in text for p in _CONTEXT_WINDOW_PATTERNS):
        return "context_window"
    if any(p in text for p in _TRANSIENT_PATTERNS):
        return "transient"
    if isinstance(exc, ProcessError):
        return "permanent"
    return "permanent"


def _delay(attempt: int) -> float:
    """Exponential backoff with up-to-25% positive jitter."""
    base = _BACKOFFS_S[min(attempt - 1, len(_BACKOFFS_S) - 1)]
    return base + random.uniform(0.0, base * 0.25)


async def with_retry(factory: Callable[[], Awaitable[T]]) -> T:
    """Run ``factory()`` with transient-error retries.

    ``factory`` must return a fresh awaitable on each call — the SDK's async
    iterator is single-use, so each retry needs a new ``query()`` invocation.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return await factory()
        except Exception as exc:
            if classify(exc) != "transient" or attempt >= MAX_ATTEMPTS:
                raise
            wait = _delay(attempt)
            ip.emit(
                f"[retry] attempt {attempt}/{MAX_ATTEMPTS - 1} failed "
                f"({type(exc).__name__}: {exc}); sleeping {wait:.1f}s"
            )
            await asyncio.sleep(wait)
    raise AssertionError("unreachable")  # for type-checkers

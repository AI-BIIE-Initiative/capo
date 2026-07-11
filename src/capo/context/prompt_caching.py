"""Prompt caching helpers for AgentRunner.

Anthropic's API supports cache_control={"type": "ephemeral"} markers on
content blocks to amortize tokens across calls within a 5-minute TTL.
claude_agent_sdk does not expose those markers directly: when the prompt
is a string it ships a single text block, and the CLI auto-caches only the
--system-prompt flag.

The streaming-mode form of query accepts an AsyncIterable[dict] and
forwards each dict as JSON to the CLI's stdin. That gives us exactly enough
freedom to build a multi-block user message where the stable prefix carries
cache_control and the mutable tail does not.

Usage stats land on AssistantMessage.usage and ResultMessage.usage as
plain dicts populated by the CLI; cache_read_input_tokens and
cache_creation_input_tokens are present whenever caching takes effect.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any


async def cached_streaming_prompt(
    stable: str, mutable: str
) -> AsyncIterator[dict[str, Any]]:
    """Yield one user-message dict in claude_agent_sdk's streaming format.

    Content is two text blocks: stable carries an ephemeral cache
    breakpoint, mutable does not. When mutable is empty, only the
    stable block is emitted.
    """
    blocks: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": stable,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    if mutable:
        blocks.append({"type": "text", "text": mutable})

    yield {
        "type": "user",
        "session_id": "",
        "message": {"role": "user", "content": blocks},
        "parent_tool_use_id": None,
    }


def extract_cache_stats(raw_messages: list[Any]) -> dict[str, int]:
    """Return the token/cache counters for one agent query.

    Both message kinds the Claude Code CLI emits carry a usage dict, but
    they mean different things:

    * AssistantMessage.usage: the usage of that single turn's API call.
      Summed across the query's turns, these already total the whole session.
    * ResultMessage.usage: the session cumulative total (the same
      source as ResultMessage.total_cost_usd).

    Adding every message's usage therefore counts the session roughly twice
    (per-turn sum + cumulative), which is exactly what can inflate the token 
    count while the SDK-reported cost stayed correct. We treat the result message 
    as authoritative, its cumulative usage lines up with the cost we record and 
    fall back to summing the per-turn assistant usages only when no result usage 
    is present (older CLI versions). Missing keys count as 0 so this is safe when 
    cache fields aren't reported.
    """

    def _blank() -> dict[str, int]:
        return {"cache_read": 0, "cache_creation": 0, "input": 0, "output": 0}

    def _accumulate(acc: dict[str, int], usage: dict[str, Any]) -> None:
        acc["cache_read"] += int(usage.get("cache_read_input_tokens") or 0)
        acc["cache_creation"] += int(usage.get("cache_creation_input_tokens") or 0)
        acc["input"] += int(usage.get("input_tokens") or 0)
        acc["output"] += int(usage.get("output_tokens") or 0)

    result_totals = _blank()
    turn_totals = _blank()
    saw_result_usage = False

    for msg in raw_messages:
        usage = getattr(msg, "usage", None)
        if not isinstance(usage, dict):
            continue
        # The result message is the only usage-carrying message that also
        # reports total_cost_usd / num_turns. its usage is the cumulative
        # session total. Everything else is a per-turn assistant slice.
        if hasattr(msg, "total_cost_usd") or hasattr(msg, "num_turns"):
            saw_result_usage = True
            _accumulate(result_totals, usage)
        else:
            _accumulate(turn_totals, usage)

    return result_totals if saw_result_usage else turn_totals

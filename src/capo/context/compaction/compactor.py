"""
Context compactor — case file + rolling tail.

The Compactor is a small async helper that:

1. Decides whether the just-finished agent call has accumulated enough
   context to warrant compaction (input-token threshold).
2. Splits the raw message list into a "rolling tail" (kept verbatim, not
   summarized) and an "older" portion (folded into a structured case file).
3. Dedupes redundant tool calls in the older portion.
4. Calls a Haiku summarizer with a tightly tuned prompt, parses its strict
   JSON output, and merges it with any prior case file.
5. Persists the new case file + appends a CompactionEvent to the run's
   compaction/history.jsonl.

The compactor is intentionally side-effecting (it writes to disk and emits
progress lines) but its core decision logic and message formatting are
pure functions, exercised directly by the unit tests.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from claude_agent_sdk import ClaudeAgentOptions, query
from claude_agent_sdk.types import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from capo.context.compaction.prompts import COMPACTION_SYSTEM_PROMPT, build_user_message
from capo.context.compaction.store import CompactionStore
from capo.context.compaction.types import (
    CaseFile,
    CompactionConfig,
    CompactionEvent,
    merge_case_file,
)
from capo.observability import progress as ip


# Loose token estimate: ~4 chars per token. Used for before/after metrics
# on summarized text where we don't have a billed token count from the API.
_CHARS_PER_TOKEN = 4

# Bash output beyond this many chars rarely carries durable signal once
# the key facts have been extracted; we trim aggressively when formatting
# tool results for the summarizer.
_MAX_TOOL_RESULT_CHARS = 600


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _estimate_tokens(text: str) -> int:
    return max(0, len(text) // _CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# Message formatting (pure)
# ---------------------------------------------------------------------------


def format_messages_for_summarizer(messages: list[Any]) -> str:
    """Render a list of SDK messages as compact text for the summarizer.

    Mirrors the human-readable form ProgressEmitter writes to run.log,
    so the summarizer sees the same view an operator would see when
    reading the run logs.
    """
    out: list[str] = []
    for msg in messages:
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    text = (block.text or "").strip()
                    if text:
                        out.append(f"[assistant] {text}")
                elif isinstance(block, ToolUseBlock):
                    args = json.dumps(block.input, ensure_ascii=False, default=str)
                    if len(args) > _MAX_TOOL_RESULT_CHARS:
                        args = args[:_MAX_TOOL_RESULT_CHARS] + "…"
                    out.append(f"[tool_call] {block.name} {args}")
                # ThinkingBlock and other variants are intentionally skipped:
                # they are not durable signal.
        elif isinstance(msg, UserMessage):
            content = msg.content
            if isinstance(content, str):
                text = content.strip()
                if text:
                    out.append(f"[user] {text}")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, ToolResultBlock):
                        body = block.content
                        if isinstance(body, list):
                            # SDK occasionally wraps tool output as a list of
                            # {"type":"text","text":...} dicts.
                            text = "\n".join(
                                str(b.get("text", "")) for b in body if isinstance(b, dict)
                            )
                        else:
                            text = str(body or "")
                        text = text.strip()
                        if len(text) > _MAX_TOOL_RESULT_CHARS:
                            text = text[:_MAX_TOOL_RESULT_CHARS] + "…"
                        marker = " [error]" if block.is_error else ""
                        out.append(f"[tool_result{marker}] {text}")
                    elif isinstance(block, TextBlock):
                        text = (block.text or "").strip()
                        if text:
                            out.append(f"[user] {text}")
        elif isinstance(msg, ResultMessage):
            # Terminal cost line — not durable, skip.
            continue
        # SystemMessage and any other variants: skip.
    return "\n".join(out)


def dedupe_tool_results(messages: list[Any]) -> list[Any]:
    """Drop superseded tool calls from a message list.

    A tool call is "superseded" when a later message in the list invokes
    the same tool with the same input. Only the LAST such call (and its
    paired result) is kept. This is the cheapest precision win — repeated
    `Bash("nvidia-smi ...")` or `Read("infra.json")` calls dominate the
    older portion of long agent traces.

    We dedupe at the level of (tool_name, json(input)) and keep the order
    of the surviving messages. Non-tool messages pass through unchanged.
    """
    last_idx_by_key: dict[tuple[str, str], int] = {}
    # First pass: find the last occurrence of each (name, input_json).
    for i, msg in enumerate(messages):
        if not isinstance(msg, AssistantMessage):
            continue
        for block in msg.content:
            if isinstance(block, ToolUseBlock):
                try:
                    key = (block.name, json.dumps(block.input, sort_keys=True, default=str))
                except (TypeError, ValueError):
                    key = (block.name, str(block.input))
                last_idx_by_key[key] = i

    # Build a set of tool_use_ids to drop (every occurrence except the last).
    drop_tool_use_ids: set[str] = set()
    for key, last_i in last_idx_by_key.items():
        for i, msg in enumerate(messages):
            if i == last_i or not isinstance(msg, AssistantMessage):
                continue
            for block in msg.content:
                if isinstance(block, ToolUseBlock):
                    try:
                        k = (block.name, json.dumps(block.input, sort_keys=True, default=str))
                    except (TypeError, ValueError):
                        k = (block.name, str(block.input))
                    if k == key:
                        drop_tool_use_ids.add(block.id)

    if not drop_tool_use_ids:
        return list(messages)

    # Second pass: rebuild the message list, dropping the superseded
    # tool_use blocks AND their paired tool_result blocks.
    out: list[Any] = []
    for msg in messages:
        if isinstance(msg, AssistantMessage):
            kept_blocks = [
                b for b in msg.content
                if not (isinstance(b, ToolUseBlock) and b.id in drop_tool_use_ids)
            ]
            if not kept_blocks:
                continue
            out.append(AssistantMessage(
                content=kept_blocks,
                model=msg.model,
                parent_tool_use_id=msg.parent_tool_use_id,
                error=msg.error,
                usage=msg.usage,
            ))
        elif isinstance(msg, UserMessage) and isinstance(msg.content, list):
            kept_blocks = [
                b for b in msg.content
                if not (isinstance(b, ToolResultBlock) and b.tool_use_id in drop_tool_use_ids)
            ]
            if not kept_blocks:
                continue
            out.append(UserMessage(
                content=kept_blocks,
                uuid=msg.uuid,
                parent_tool_use_id=msg.parent_tool_use_id,
                tool_use_result=msg.tool_use_result,
            ))
        else:
            out.append(msg)
    return out


# ---------------------------------------------------------------------------
# Compactor
# ---------------------------------------------------------------------------


# Type alias for the summarizer call. Tests inject a fake; production uses
# `_default_summarizer` which calls claude_agent_sdk.query.
SummarizerFn = Callable[[str, str, str], Awaitable[tuple[str, float | None]]]


async def _default_summarizer(
    system_prompt: str,
    user_message: str,
    model: str,
) -> tuple[str, float | None]:
    """Call Haiku once, with no tools, and return (raw_text, cost_usd).

    The summarizer is a single non-tool roundtrip — running it through
    AgentRunner would add unnecessary subagent / MCP machinery.
    """
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=system_prompt,
        allowed_tools=[],
        permission_mode="default",
        max_turns=1,
    )
    parts: list[str] = []
    cost: float | None = None
    async for msg in query(prompt=user_message, options=options):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock) and block.text:
                    parts.append(block.text)
        elif isinstance(msg, ResultMessage):
            cost = msg.total_cost_usd
    return "".join(parts), cost


class Compactor:
    """Decides when to compact and produces a CaseFile."""

    def __init__(
        self,
        config: CompactionConfig,
        *,
        run_id: str,
        local_run_dir: Path | str,
        summarizer_model: str = "claude-haiku-4-5-20251001",
        summarizer: SummarizerFn | None = None,
    ) -> None:
        self._cfg = config
        self._run_id = run_id
        self._store = CompactionStore(local_run_dir)
        self._model = summarizer_model
        self._summarizer = summarizer or _default_summarizer

    # ------------------------------------------------------------------ #
    # Public                                                              #
    # ------------------------------------------------------------------ #

    def should_compact(
        self,
        *,
        last_input_tokens: int,
        n_messages: int,
    ) -> bool:
        """Pure predicate — exposed so callers (and tests) can verify."""
        if not self._cfg.enabled:
            return False
        if last_input_tokens < self._cfg.threshold_input_tokens:
            return False
        older_count = max(0, n_messages - self._cfg.keep_recent_messages)
        return older_count >= self._cfg.min_older_messages

    async def maybe_compact(
        self,
        *,
        phase_label: str,
        raw_messages: list[Any],
        prior_case_file: CaseFile | None,
        last_input_tokens: int,
    ) -> CaseFile | None:
        """Run compaction iff the threshold is crossed.

        Returns the new (merged) case file on success, or None if the
        threshold wasn't met or summarization failed in a way we couldn't
        recover from. Failures are logged but never raised — compaction
        is best-effort and must never abort a run.
        """
        if not self.should_compact(
            last_input_tokens=last_input_tokens,
            n_messages=len(raw_messages),
        ):
            return None

        keep = self._cfg.keep_recent_messages
        older = raw_messages[:-keep] if keep > 0 else list(raw_messages)
        tail = raw_messages[-keep:] if keep > 0 else []

        deduped = dedupe_tool_results(older)
        older_text = format_messages_for_summarizer(deduped)
        tail_text = format_messages_for_summarizer(tail)

        if not older_text.strip():
            ip.emit("[compaction] older portion is empty after dedup — skipping.")
            return None

        tokens_before = _estimate_tokens(older_text)

        prior_md = prior_case_file.to_markdown() if prior_case_file else None
        user_message = build_user_message(
            prior_case_file_markdown=prior_md,
            older_messages_text=older_text,
            rolling_tail_summary=tail_text,
        )

        ip.emit(
            f"[compaction] {phase_label}: summarizing {len(deduped)} messages "
            f"(~{tokens_before:,} tokens, threshold {self._cfg.threshold_input_tokens:,})"
        )

        try:
            raw, cost = await self._summarizer(
                COMPACTION_SYSTEM_PROMPT, user_message, self._model
            )
        except Exception as exc:  # pragma: no cover - defensive
            ip.error(f"[compaction] summarizer call failed: {exc}")
            return None

        additions = _parse_summary_json(raw)
        if additions is None:
            ip.error(
                "[compaction] summarizer returned unparseable output; "
                "keeping prior case file unchanged."
            )
            return None

        # Apply char cap defensively in case the model exceeded it.
        if isinstance(additions.get("narrative"), str):
            additions["narrative"] = additions["narrative"][: self._cfg.max_summary_chars]

        tokens_after = _estimate_tokens(json.dumps(additions, ensure_ascii=False))
        new_case = merge_case_file(
            prior_case_file,
            run_id=self._run_id,
            additions=additions,
            metrics_delta=(len(deduped), tokens_before, tokens_after),
        )

        try:
            self._store.save_case_file(new_case)
        except OSError as exc:
            ip.error(f"[compaction] failed to persist case file: {exc}")
            # Still return the in-memory case file so the caller can use it
            # this turn; the next compaction will retry persistence.

        reduction = (
            (1.0 - tokens_after / tokens_before) if tokens_before > 0 else 0.0
        )
        event = CompactionEvent(
            timestamp=_now_iso(),
            run_id=self._run_id,
            phase_label=phase_label,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            reduction_pct=reduction,
            messages_summarized=len(deduped),
            rolling_tail_kept=len(tail),
            summarizer_cost_usd=cost,
        )
        try:
            self._store.append_event(event)
        except OSError as exc:  # pragma: no cover - defensive
            ip.error(f"[compaction] failed to append history event: {exc}")

        ip.emit(
            f"[compaction] done: ~{tokens_before:,} → ~{tokens_after:,} tokens "
            f"({reduction:.1%} reduction), tail={len(tail)} messages kept verbatim"
        )
        return new_case


class NullCompactor(Compactor):
    """No-op compactor for callers that want unconditional non-compaction.

    Lets the orchestrator hold a single typed reference instead of a
    `Compactor | None`, keeping call sites free of if self._compactor:.
    """

    def __init__(self) -> None:  # noqa: D401 - intentional override
        # Skip parent __init__ — no config, no store, no summarizer needed.
        self._cfg = CompactionConfig(enabled=False)

    def should_compact(self, **_: Any) -> bool:
        return False

    async def maybe_compact(self, **_: Any) -> CaseFile | None:
        return None


# ---------------------------------------------------------------------------
# JSON parsing (tolerant)
# ---------------------------------------------------------------------------


_REQUIRED_KEYS = {
    "decisions",
    "constraints",
    "file_findings",
    "open_questions",
    "errors_resolved",
    "artifacts_produced",
    "narrative",
}


def _parse_summary_json(text: str) -> dict | None:
    """Best-effort JSON parse + shape check.

    Tolerates fences and surrounding prose. Returns None when the output
    isn't a dict containing AT LEAST one of the required keys — anything
    less is a sign the model lost the schema entirely and we shouldn't
    overwrite the prior case file from it.
    """
    if not text:
        return None

    candidates: list[str] = [text.strip()]
    # Strip ```json fences if present.
    stripped = text.strip()
    if stripped.startswith("```"):
        # Remove first fence line and trailing fence.
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        candidates.append(stripped.strip())
    # Slice from first { to last }.
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last > first:
        candidates.append(text[first : last + 1])

    for c in candidates:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if _REQUIRED_KEYS & set(obj.keys()):
            return obj
    return None

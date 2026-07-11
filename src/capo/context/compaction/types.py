"""
Dataclasses for context compaction.

The compaction subsystem builds a small, durable "case file" that carries
the irreducible facts of a run forward across phase boundaries. The case
file replaces a long verbatim message history with structured sections the
next agent invocation can rely on.

A separate CompactionEvent records what happened on each compaction
(tokens before/after, messages summarized, cost) for observability.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone


_SCHEMA_VERSION = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompactionConfig:
    """
    Knobs controlling when and how compaction runs.

    Defaults are tuned for the FineTuningOrchestrator's Sonnet pre-launch
    phase, which routinely accumulates ~80-120k input tokens across its
    20+ internal turns.
    """

    enabled: bool = True

    # Trigger when the just-finished call's billed input tokens
    # (cache_read + cache_creation + input) cross this threshold
    threshold_input_tokens: int = 80_000

    # Number of most-recent raw messages to preserve verbatim as the
    # rolling tail. The rest are folded into the case file
    keep_recent_messages: int = 5

    # Minimum number of older messages required for compaction to run.
    # Below this, summarization is unlikely to recoup its own cost
    min_older_messages: int = 8

    # Cap on the JSON the summarizer is allowed to emit. The model is
    # instructed to follow this, also truncate defensively on save
    max_summary_chars: int = 24_000


# ---------------------------------------------------------------------------
# Case file (durable summary)
# ---------------------------------------------------------------------------


@dataclass
class CaseFile:
    """
    The structured, durable summary carried across phases.

    Each list field accumulates monotonically across compactions: a later
    compaction merges its own findings with the prior case file, never
    discards them. The free-form narrative is rewritten in full each
    time so it stays bounded.
    """

    run_id: str
    updated_at: str
    schema_version: int = _SCHEMA_VERSION

    # Structured durable content
    decisions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    file_findings: dict[str, str] = field(default_factory=dict)
    open_questions: list[str] = field(default_factory=list)
    errors_resolved: list[str] = field(default_factory=list)
    artifacts_produced: list[str] = field(default_factory=list)
    narrative: str = ""

    # Cumulative observability metrics (across all compactions for this run)
    cumulative_messages_summarized: int = 0
    cumulative_tokens_before: int = 0
    cumulative_tokens_after: int = 0
    compactions: int = 0

    # ---------------------------------------------------------------------------
    # Serialization                                                       
    # ---------------------------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=False, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "CaseFile":
        payload = json.loads(text)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in known})

    def to_markdown(self) -> str:
        """Render as a compact markdown block for injection into prompts.

        Empty sections are omitted so the output stays terse when little
        durable context exists.
        """
        out: list[str] = [
            f"# Compacted prior context (run {self.run_id})",
            f"_updated {self.updated_at} · {self.compactions} compaction(s) · "
            f"~{self.cumulative_tokens_before:,} → ~{self.cumulative_tokens_after:,} tokens_",
            "",
        ]

        def _section(title: str, items: list[str]) -> None:
            if not items:
                return
            out.append(f"## {title}")
            for item in items:
                out.append(f"- {item}")
            out.append("")

        _section("Decisions", self.decisions)
        _section("Constraints", self.constraints)
        if self.file_findings:
            out.append("## Files visited")
            for path, summary in sorted(self.file_findings.items()):
                out.append(f"- `{path}` — {summary}")
            out.append("")
        _section("Artifacts produced", self.artifacts_produced)
        _section("Errors resolved", self.errors_resolved)
        _section("Open questions", self.open_questions)
        if self.narrative.strip():
            out.append("## Narrative")
            out.append(self.narrative.strip())
            out.append("")
        return "\n".join(out).rstrip() + "\n"


def merge_case_file(
    prior: CaseFile | None,
    *,
    run_id: str,
    additions: dict,
    metrics_delta: tuple[int, int, int],
) -> CaseFile:
    """Merge fresh summarizer output into the prior case file.

    `additions` is the parsed JSON the summarizer emitted (already
    schema-validated by the caller). Lists are concatenated and deduped
    while preserving order; the dict of file findings is merged with later
    summaries overriding earlier ones for the same path; the narrative is
    replaced wholesale (the new one is meant to subsume the old).

    `metrics_delta` is `(messages_summarized, tokens_before, tokens_after)`
    for this single compaction; we accumulate it onto the prior totals.
    """
    base = prior or CaseFile(run_id=run_id, updated_at=_now_iso())

    def _extend_unique(existing: list[str], new: list) -> list[str]:
        out = list(existing)
        seen = set(existing)
        for item in new or []:
            s = str(item).strip()
            if s and s not in seen:
                out.append(s)
                seen.add(s)
        return out

    decisions = _extend_unique(base.decisions, additions.get("decisions") or [])
    constraints = _extend_unique(base.constraints, additions.get("constraints") or [])
    open_questions = _extend_unique(base.open_questions, additions.get("open_questions") or [])
    errors_resolved = _extend_unique(base.errors_resolved, additions.get("errors_resolved") or [])
    artifacts_produced = _extend_unique(
        base.artifacts_produced, additions.get("artifacts_produced") or []
    )

    file_findings = dict(base.file_findings)
    raw_findings = additions.get("file_findings") or {}
    if isinstance(raw_findings, dict):
        for path, summary in raw_findings.items():
            if isinstance(path, str) and isinstance(summary, str):
                file_findings[path] = summary.strip()

    narrative = str(additions.get("narrative") or base.narrative or "").strip()

    msgs, tok_before, tok_after = metrics_delta
    return CaseFile(
        run_id=run_id,
        updated_at=_now_iso(),
        schema_version=_SCHEMA_VERSION,
        decisions=decisions,
        constraints=constraints,
        file_findings=file_findings,
        open_questions=open_questions,
        errors_resolved=errors_resolved,
        artifacts_produced=artifacts_produced,
        narrative=narrative,
        cumulative_messages_summarized=base.cumulative_messages_summarized + msgs,
        cumulative_tokens_before=base.cumulative_tokens_before + tok_before,
        cumulative_tokens_after=base.cumulative_tokens_after + tok_after,
        compactions=base.compactions + 1,
    )


# ---------------------------------------------------------------------------
# per-event observability record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompactionEvent:
    """One row appended to compaction/history.jsonl per compaction."""

    timestamp: str
    run_id: str
    phase_label: str
    tokens_before: int
    tokens_after: int
    reduction_pct: float
    messages_summarized: int
    rolling_tail_kept: int
    summarizer_cost_usd: float | None

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

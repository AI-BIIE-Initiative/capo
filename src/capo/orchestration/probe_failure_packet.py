"""
probe_failure_packet.py — Compact failure packet for the 3-step gate repair ladder.

The repair ladder (gate.py) passes only this packet to self-repair turns and to
the code-repair-critic subagent. Keeping it small is the token-efficiency lever
that lets the ladder beat today's full-prompt repair loop.

Contract:
  build_compact_packet(...) returns a JSON-serialisable dict ≤ 8 KB.
  Truncation order when over budget: history first (keep last 2 attempts only),
  then traceback (shrink to last 40 lines), then failing-file excerpt.

Used by:
  - src/capo/orchestration/gate.py     — builds packet on probe failure
  - src/capo/orchestration/agent_runner.py (code-repair-critic) — sole input
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_MAX_PACKET_BYTES = 8 * 1024
_DEFAULT_TRACEBACK_LINES = 80
_FALLBACK_TRACEBACK_LINES = 40
_DIFF_HEADER_RE = re.compile(r"^(?:diff --git |--- |\+\+\+ )", re.MULTILINE)


def redact_traceback(traceback: str, max_lines: int = _DEFAULT_TRACEBACK_LINES) -> str:
    """Keep the last `max_lines` lines of a traceback — that's where the cause lives."""
    if not traceback:
        return ""
    lines = traceback.splitlines()
    if len(lines) <= max_lines:
        return traceback.rstrip()
    head = f"... [{len(lines) - max_lines} earlier lines truncated]"
    return "\n".join([head, *lines[-max_lines:]]).rstrip()


def _read_file_excerpt(path: Path, max_lines: int = 200) -> str:
    """Best-effort excerpt of the failing file. Empty string if unreadable."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join([*lines[: max_lines // 2], "... [middle truncated]", *lines[-max_lines // 2 :]])


def build_compact_packet(
    *,
    failing_file: str,
    failure_category: str,
    traceback: str,
    expected_schema: dict[str, Any],
    budget: dict[str, Any],
    code_spec_ref: str = "code_spec.json",
    history: list[dict[str, Any]] | None = None,
    failing_file_root: Path | None = None,
) -> dict[str, Any]:
    """Build the compact failure packet shared by orchestrator self-repair and the critic.

    Args:
        failing_file: Path relative to the run dir (e.g. "probe.py" or a module under src/).
        failure_category: One of {script_bug, data_schema_mismatch, resource_mismatch, oom, nan_inf}.
        traceback: Full stderr / probe.log traceback. Will be truncated.
        expected_schema: Task/label schema the file must satisfy (label column, task type, etc).
        budget: vram_gb, max_cost_usd, hourly_rate_usd.
        code_spec_ref: Pointer to the code_spec.json the orchestrator wrote at Attempt 0.
        history: Prior attempt records, each {"attempt": int, "summary": str, "diff_path": str}.
        failing_file_root: If set, include a 200-line excerpt of failing_file from this root.

    Returns:
        A dict serialisable via json.dumps; size ≤ 8 KB.
    """
    history = history or []

    packet: dict[str, Any] = {
        "failing_file": failing_file,
        "failure_category": failure_category,
        "traceback": redact_traceback(traceback, _DEFAULT_TRACEBACK_LINES),
        "expected_schema": expected_schema,
        "budget": budget,
        "code_spec_ref": code_spec_ref,
        "history": history,
    }

    if failing_file_root is not None:
        excerpt = _read_file_excerpt(failing_file_root / failing_file)
        if excerpt:
            packet["failing_file_excerpt"] = excerpt

    # Enforce the 8 KB cap. Truncate in this order:
    #   1. history → keep last 2 attempts only
    #   2. traceback → re-redact to 40 lines
    #   3. failing_file_excerpt → drop entirely
    if len(json.dumps(packet)) > _MAX_PACKET_BYTES and len(packet["history"]) > 2:
        packet["history"] = packet["history"][-2:]

    if len(json.dumps(packet)) > _MAX_PACKET_BYTES:
        packet["traceback"] = redact_traceback(traceback, _FALLBACK_TRACEBACK_LINES)

    if len(json.dumps(packet)) > _MAX_PACKET_BYTES and "failing_file_excerpt" in packet:
        del packet["failing_file_excerpt"]

    return packet


def validate_diff(diff_text: str) -> bool:
    """Cheap structural check that `diff_text` looks like a unified diff.

    The critic's output must apply via `git apply`. We don't replicate that here —
    just reject obviously non-diff output (empty, no headers, plain prose).
    """
    if not diff_text or not diff_text.strip():
        return False
    return bool(_DIFF_HEADER_RE.search(diff_text))


def write_packet(packet: dict[str, Any], out_path: Path) -> Path:
    """Write the packet to disk as JSON (pretty-printed). Returns out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(packet, indent=2), encoding="utf-8")
    return out_path

"""
Tail outputs/run.log and re-render each line cleanly and colour-coded.

The orchestrator's ProgressEmitter writes lines as HH:MM:SS [tag] message,
and during the parallel pre-launch phase each line is additionally prefixed
with a runner tag, e.g. HH:MM:SS [infra-agent] [lambda] message. Multi-line
agent narration ([status] ...) spills onto continuation lines that carry no
timestamp.

This module turns that stream into something readable:

  • the first bracket is treated as the SOURCE (infra / data / model / research /
    memory / trackio / health) and rendered as one small colour-coded chip — the
    orchestrator itself stays white, with no chip;
  • the redundant inner [event] tag is dropped (the message is descriptive),
    keeping only semantic colour for warnings / errors / progress / success;
  • each line is a borderless 2-column grid, so a long message wraps with a
    hanging indent under the message column instead of falling back to column 0;
  • continuation lines are folded under their parent, indented and dimmed,
    instead of appearing as a raw wall of un-timestamped text;
  • a blank line is inserted before each major phase so the log reads in
    sections rather than one dense block.

Nothing in the orchestrator changes — this is pure presentation over the log
files it already writes.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from rich.table import Table
from rich.text import Text

from .colors import console

# HH:MM:SS [tag] message — tag may contain a . (e.g. ssh.cmd) or -.
_LINE_RE = re.compile(r"^(?P<ts>\d{2}:\d{2}:\d{2})\s+\[(?P<tag>[^\]]+)\]\s*(?P<rest>.*)$")
# a leading inner [event] ... inside the remainder of a sourced line.
_INNER_RE = re.compile(r"^\[(?P<tag>[^\]]+)\]\s*(?P<msg>.*)$")

# first bracket → (chip label, source style). Anything not here is the
# orchestrator (white, no chip) and its first bracket is an event tag.
_SOURCES: dict[str, tuple[str, str]] = {
    "infra-agent": ("infra", "src.infra"),
    "infra": ("infra", "src.infra"),
    "data-agent": ("data", "src.data"),
    "data": ("data", "src.data"),
    "model-sel-agent": ("model", "src.model"),
    "model-agent": ("model", "src.model"),
    "research": ("research", "src.research"),
    "memory": ("memory", "src.memory"),
    "trackio": ("trackio", "src.tracker"),
    "health": ("health", "src.health"),
    "training-health-monitor": ("health", "src.health"),
    "finalizer": ("finalize", "src.orchestrator"),
}

# left (fixed) column width: "  " + ts(8) + "  " + chip(8) = 20; message follows.
_PREFIX_W = 20
_CHIP_W = 8
_NOISE_TRUNC = 100

# agent reasoning / narration tags → subtle light-grey-italic style 
_REASONING_TAGS: frozenset[str] = frozenset(
    {"status", "reasoning", "thinking", "thought", "plan", "rationale", "note"}
)

# major-phase cues — a blank line is inserted before a line matching any of these.
_SECTION_RE = re.compile(
    r"(Phase[s]?\s*\d|parallel pre-launch|Starting pre-pipeline|Cost gate|"
    r"cost gate|\bGATE\b|Launching training|training launched|Handing off|"
    r"Finaliz|started_at=)",
)


def _tag_style(tag: str) -> str:
    """Legacy tag → theme style (kept for callers/tests; longest-prefix match)."""
    table = {
        "lambda": "tag.lambda", "ssh": "tag.lambda", "tmux": "tag.tmux",
        "rsync": "tag.rsync", "setup": "tag.setup", "fine-tuning": "tag.training",
        "inference": "tag.training", "training": "tag.training", "agent": "tag.agent",
        "research": "tag.agent", "memory": "tag.agent", "shell": "tag.shell",
        "hardware": "tag.hardware", "results": "tag.results", "summary": "tag.summary",
        "progress": "tag.training",
    }
    if tag.endswith(".cmd"):
        return "tag.default"
    best = ""
    for prefix in table:
        if tag.startswith(prefix) and len(prefix) > len(best):
            best = prefix
    return table.get(best, "tag.default")


def _clean_md(s: str) -> str:
    """Lightly de-markdown agent narration so it reads as plain log text."""
    s = s.replace("**", "")
    s = re.sub(r"^#{1,6}\s+", "", s)          # strip heading markers
    s = re.sub(r"^[-*]\s+", "• ", s)          # bullets → •
    return s


def _msg_style(event_tag: str | None, msg: str) -> tuple[str | None, bool]:
    """Return (rich style for the message, is_noise). None style = default white."""
    tag = (event_tag or "").lower()
    if tag.endswith(".cmd") or tag in ("cache", "compaction"):
        return "log.noise", True
    if msg.startswith("Loading tool schema"):
        return "log.noise", True              # mcp schema dumps — truncate
    if tag in ("error", "anthropic-auth", "agent-exit", "agent-answer", "cli"):
        return "log.err", False
    if tag == "warning" or msg.lower().startswith("warning"):
        return "log.warn", False
    if tag == "summary":
        return "log.ok", False
    if tag == "progress":
        return "log.progress", False
    if tag in _REASONING_TAGS:
        return "log.cont", False             
    return None, False


@dataclass
class _Parsed:
    ts: str
    chip: str
    chip_style: str
    msg: str
    msg_style: str | None


def _parse_line(raw: str, error: bool = False) -> _Parsed | None:
    """Parse a full HH:MM:SS [tag] ... line, or None if it isn't one."""
    m = _LINE_RE.match(raw.rstrip())
    if not m:
        return None
    ts, tag1, rest = m.group("ts"), m.group("tag"), m.group("rest")

    if tag1 in _SOURCES:
        chip, chip_style = _SOURCES[tag1]
        inner = _INNER_RE.match(rest)
        event_tag, msg = (inner.group("tag"), inner.group("msg")) if inner else (None, rest)
    else:
        chip, chip_style = "", ""             # orchestrator: white, no chip
        event_tag, msg = tag1, rest

    msg = _clean_md(msg.strip())
    style, is_noise = _msg_style(event_tag, msg)
    if error and style is None:
        style = "log.err"
    if is_noise and len(msg) > _NOISE_TRUNC:
        msg = msg[:_NOISE_TRUNC] + "…"
    return _Parsed(ts=ts, chip=chip, chip_style=chip_style, msg=msg, msg_style=style)


def _prefix_text(p: _Parsed) -> Text:
    """The fixed-width left column: '  HH:MM:SS  <chip>'."""
    t = Text("  ")
    t.append(f"{p.ts:<8}", style="log.ts")
    t.append("  ")
    if p.chip:
        t.append(f"{p.chip:<{_CHIP_W}}", style=p.chip_style)
    else:
        t.append(" " * _CHIP_W)
    return t


def _grid(left: Text, msg: Text) -> Table:
    """A borderless 2-col grid → message wraps with a hanging indent."""
    g = Table.grid(padding=(0, 1, 0, 0))
    g.add_column(width=_PREFIX_W, no_wrap=True)
    g.add_column(overflow="fold", ratio=1)
    g.add_row(left, msg)
    return g


def render_line(raw: str) -> Text:
    """Stateless single-line render (used in tests). Non-matching → verbatim."""
    p = _parse_line(raw)
    if p is None:
        return Text(raw.rstrip(), style="muted")
    t = _prefix_text(p)
    t.append(" ")
    t.append(p.msg, style=p.msg_style)
    return t


class _LogRenderer:
    """Stateful renderer: folds continuation lines and spaces out phases."""

    def __init__(self, error: bool = False) -> None:
        self._error = error
        self._emitted = False
        self._last_blank = True               # suppress a leading blank line

    def feed(self, raw: str):
        """Render one raw log line into 0–2 renderables (incl. optional spacer)."""
        line = raw.rstrip("\n")
        out: list = []

        p = _parse_line(line, error=self._error)
        if p is None:
            # continuation of a multi-line message (or a blank line in the log).
            text = _clean_md(line.strip())
            if not text:
                return out
            self._last_blank = False
            return [_grid(Text(""), Text(text, style="log.err" if self._error else "log.cont"))]

        # a real, timestamped line — insert a phase spacer if warranted.
        if not self._error and self._emitted and not self._last_blank and _SECTION_RE.search(line):
            out.append(Text(""))
            self._last_blank = True

        if not p.msg:
            return out
        out.append(_grid(_prefix_text(p), Text(p.msg, style=p.msg_style)))
        self._emitted = True
        self._last_blank = False
        return out


def stream_log(
    log_path: Path,
    stop_event: threading.Event,
    from_start: bool = False,
    error: bool = False,
    sink: Callable[[object], None] | None = None,
) -> None:
    """
    Tail log_path, rendering lines until stop_event is set.

    from_start=True reads from byte 0 — use for a fresh run so the first lines
    emitted in the race between file creation and attach are never missed.
    from_start=False seeks to the current end — use on resume so the prior log
    is not replayed. error=True renders the stderr stream in the error style.

    sink receives each rendered Rich renderable. It defaults to printing on the
    shared console (the plain run path and every test). the full-screen RunConsole
    passes its own sink to capture the renderable into its log pane instead, so
    there is exactly one terminal writer during a run.
    """
    renderer = _LogRenderer(error=error)
    _sink = sink if sink is not None else (lambda piece: console.print(piece, highlight=False))

    def emit(line: str) -> None:
        for piece in renderer.feed(line):
            _sink(piece)

    # wait for the file to appear (created a few seconds into the run).
    while not stop_event.is_set() and not log_path.exists():
        time.sleep(0.2)
    if not log_path.exists():
        return

    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        if not from_start:
            fh.seek(0, 2)  # start at end — only show new lines
        while not stop_event.is_set():
            line = fh.readline()
            if line:
                emit(line)
            else:
                time.sleep(0.1)
        # drain whatever remains on shutdown so a fast run is never cut off.
        for line in fh:
            emit(line)

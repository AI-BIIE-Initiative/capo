"""
The first conversational layer of the CLI: a Sonnet-backed CAPO assistant.

Bare capo enters this loop. The assistant understands what CAPO is and can do,
reasons about the user's free-text request, asks only the clarifying questions
that matter (arrow-key pickers, always with a free-text "Other…"), and decides
when the intent is clear enough to launch. It then hands a finalized, enriched
task to the orchestrator (exactly the task.md → HF-research / memory / main-agent
path that already exists). Slash commands (/help /config /history /quit) work at
any point in the conversation.

Design notes
- Each turn is a stateless Sonnet call with the whole transcript + a running
  task draft; the model replies with a strict JSON object (reply / questions /
  task / ready). _call_model and _read_user are the seams tests drive.
- The agent runner prints its own progress lines via the global emitter; during
  a chat there is none, so those fall back to bare print(). We redirect stdout
  for the duration of the model call so the chat stays clean.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import json
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.lexers import Lexer

from .colors import PURPLE_0, console
from .commands import command_help_table, is_quit, resolve_slash
from .config import CapoConfig

# slash commands recognised at the chat front door (Tab-completes / lexer).
# /health /abort /status are run-only, so they are not offered here.
_CHAT_COMMANDS = ["/help", "/config", "/history", "/prune-memory", "/tune", "/retune", "/quit"]

_TASK_KEYS = (
    "objective", "mode", "dataset_ref", "model_id",
    "fine_tune_strategy", "gpu_preference", "max_cost_usd", "notes",
    # scientific detail that shapes the structured task.md (all optional)
    "title", "organism", "target", "evaluation", "deliverables", "constraints",
)


@dataclass
class ChatPlan:
    """Resolved outcome of the chat : what app.py needs to launch."""

    objective: str
    mode: str                      # "fine-tune" | "pre-train"
    dataset_ref: str
    model_id: str
    fine_tune_strategy: str
    gpu_preference: Optional[str]
    max_cost_usd: float
    notes: str
    enriched_description: str      # → task.md (run_planner enrichment)


# ── slash-command lexer (purple only when fully typed) ────────────────────────


class _ChatLexer(Lexer):
    def lex_document(self, document):
        lines = document.lines

        def get_line(lineno: int):
            line = lines[lineno]
            first = line.lstrip().split(" ", 1)[0]
            if resolve_slash(first, _CHAT_COMMANDS)[0] is not None:
                indent = line[: len(line) - len(line.lstrip())]
                rest = line.lstrip()[len(first):]
                frags = [("class:cmd", first)]
                if indent:
                    frags.insert(0, ("", indent))
                if rest:
                    frags.append(("", rest))
                return frags
            return [("", line)]

        return get_line


# ── model I/O (the test seams) ────────────────────────────────────────────────


def _make_runner(
    cfg: CapoConfig,
    *,
    prompt_name: str = "cli/chat_assistant",
    allowed_tools: Optional[list[str]] = None,
    cwd: Optional[str] = None,
    max_turns: int = 2,
):
    """Build the chat's Sonnet runner (lazy import, SDK is heavy).

    Defaults reproduce the front-door agent exactly — no tools, repo-root cwd,
    the chat_assistant prompt. The post-run chat passes read-only file tools, the
    finished run dir as cwd, and the post_run_chat prompt so it can inspect
    artifacts while reusing the same JSON contract and launch path."""
    from capo.orchestration.agent_runner import AgentRunner

    model = cfg.model_name if cfg.model_name in AgentRunner.SUPPORTED_MODELS else "claude-sonnet-4-6"
    from capo.utils.prompts import load_prompt

    return AgentRunner(
        model_name=model,
        allowed_tools=allowed_tools or [],
        system_prompt=load_prompt(prompt_name),
        permission_mode="bypassPermissions",
        max_turns=max_turns,
        cwd=cwd or str(cfg.repo_root),
        mcp_servers={},
        emit_cost_per_call=False,
    )


def _call_model(runner, prompt: str, max_turns: int = 2) -> str:
    """Run one model turn, swallowing the runner's stdout/stderr telemetry.

    max_turns defaults to 2 (the front door's single-shot JSON reply).
    The post-run chat raises it so the agent can read a few artifacts via its file
    tools before emitting the final JSON object."""
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        res = runner.generate_sync(prompt=prompt, max_turns=max_turns)
    return res.answer or ""


# === animated "thinking" indicator ===

_C_PURPLE = "\033[1;38;2;113;61;143m"
_C_DIM = "\033[2m"
_C_RESET = "\033[0m"
_DOTS = ("", ".", "..", "...")
# after this many seconds the spinner relabels "thinking" → "still tuning" so a
# long wait reads as steady work rather than a hang
_TUNING_AFTER = 15.0


def _thinking_label(elapsed: float, base: str = "thinking") -> str:
    """Spinner verb: the base for the first _TUNING_AFTER seconds, then a one-shot
    switch to 'still tuning' so a long model call never looks frozen."""
    return base if elapsed < _TUNING_AFTER else "still tuning"


class _Thinking:
    """Animate 'CAPO is thinking…' on the real stdout while the model runs.

    _call_model redirects sys.stdout to a sink to swallow agent telemetry, so
    we capture the real stream on entry (before that redirect) and write the
    animation straight to it from a daemon thread. No-op when stdout is not a
    TTY (tests, pipes), so captured output stays clean.
    """

    def __init__(self, label: str = "thinking") -> None:
        self._label = label
        self._out = sys.stdout  # the real stream, captured before any redirect
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "_Thinking":
        if getattr(self._out, "isatty", lambda: False)():
            self._out.write("\n")  # breathing room between the user's line and the spinner
            self._out.flush()
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def _spin(self) -> None:
        import time

        t0 = time.monotonic()
        for dots in itertools.cycle(_DOTS):
            if self._stop.is_set():
                break
            # one-shot relabel after _TUNING_AFTER so a long wait reads as work.
            label = _thinking_label(time.monotonic() - t0, self._label)
            pad = " " * (4 - len(dots))  # constant width so shrinking dots clear
            self._out.write(f"\r  {_C_PURPLE}CAPO{_C_RESET}{_C_DIM} is "
                            f"{label}{dots}{_C_RESET}{pad}")
            self._out.flush()
            self._stop.wait(0.35)

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._out.write("\r" + " " * 36 + "\r")  # erase the line (fits 'still tuning…')
            self._out.flush()


def _make_session() -> PromptSession:
    return PromptSession(
        history=InMemoryHistory(),
        completer=WordCompleter(_CHAT_COMMANDS, sentence=True),
        complete_while_typing=True,
        lexer=_ChatLexer(),
    )


def _read_user(session: PromptSession) -> str:
    try:
        return session.prompt(HTML(f'<style fg="{PURPLE_0}"><b>  ❯ </b></style>')).strip()
    except (KeyboardInterrupt, EOFError):
        return "/quit"


# === JSON contract parsing ===

# appended to the prompt on a single retry when the model's first reply had no
# parseable JSON — a formatting nudge, not a change to what we're asking for.
_JSON_NUDGE = (
    "\n\nREMINDER: reply with EXACTLY ONE JSON object matching your schema — "
    "no prose before or after, no code fences."
)


def _extract_json(text: str) -> Optional[dict]:
    """Pull the single JSON object out of a model reply (tolerant of fences)."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    i, j = t.find("{"), t.rfind("}")
    if i == -1 or j == -1 or j < i:
        return None
    blob = t[i: j + 1]
    for candidate in (blob, re.sub(r",(\s*[}\]])", r"\1", blob)):  # retry w/o trailing commas
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def _salvage_reply(raw: str) -> str:
    """Recover a presentable plain-prose reply when the model answered without a
    parseable JSON object, so a clear request is never stonewalled with "I didn't
    follow". Returns "" when there's nothing clean to show (a half-formed JSON
    blob is never echoed) — the caller then asks for a rephrase."""
    t = raw.strip()
    if not t:
        return ""
    if t.startswith("```"):  # strip code fences the model may have wrapped prose in
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()
    if "{" in t or "}" in t:  # a failed JSON attempt is not user-presentable
        return ""
    return t


def _merge_draft(draft: dict, task: dict) -> None:
    """Fold a model-provided task into the running draft (non-null wins)."""
    if not isinstance(task, dict):
        return
    for k in _TASK_KEYS:
        v = task.get(k)
        if v not in (None, "", "null"):
            draft[k] = v


# === context block ===


def _recent_runs(runs_root: Path, n: int = 5) -> list[str]:
    if not runs_root.exists():
        return []
    states = sorted(runs_root.glob("*/state.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    out: list[str] = []
    for sf in states[:n]:
        try:
            d = json.loads(sf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out.append(f"{d.get('run_id', sf.parent.name)} — {d.get('dataset_ref', '?')} "
                   f"({d.get('current_phase', '?')})")
    return out


def _project_context(cfg: CapoConfig, runs_root: Path) -> str:
    runs = _recent_runs(runs_root)
    runs_block = "\n".join(f"  - {r}" for r in runs) if runs else "  - none yet"
    return (
        "PROJECT DEFAULTS (use as fallbacks; do not re-ask what is already set):\n"
        f"  dataset_ref: {cfg.dataset_ref}\n"
        f"  model_id: {cfg.model_id}\n"
        f"  fine_tune_strategy: {cfg.fine_tune_strategy}\n"
        f"  gpu_preference: {cfg.gpu_preference or 'auto'}\n"
        f"  max_cost_usd: {cfg.max_cost_usd}\n"
        f"  episodic_memory: {'on' if cfg.enable_memory else 'off'}; "
        f"hf_research: {'on' if cfg.enable_hf_research else 'off'}\n"
        "RECENT RUNS:\n"
        f"{runs_block}"
    )


def _build_prompt(cfg: CapoConfig, runs_root: Path, draft: dict, messages: list[tuple[str, str]]) -> str:
    convo = "\n".join(f"[{role}] {text}" for role, text in messages)
    return (
        f"{_project_context(cfg, runs_root)}\n\n"
        f"TASK DRAFT (gathered so far; null = unknown):\n{json.dumps(draft)}\n\n"
        f"CONVERSATION:\n{convo}\n\n"
        "Reply with ONE JSON object now (schema in your instructions)."
    )


# === questions ===


def _ask_questions(questions: list) -> dict:
    """Render each model question as an arrow-key picker / free-text input."""
    from .widgets import select_one, text_input

    answers: dict = {}
    for q in questions:
        if not isinstance(q, dict):
            continue
        key = q.get("key") or q.get("prompt", "answer")
        prompt = q.get("prompt", "?")
        choices = [str(c) for c in (q.get("choices") or [])]
        try:
            ans = select_one(prompt, choices, allow_other=True) if choices else text_input(prompt)
        except KeyboardInterrupt:
            ans = ""
        if ans:
            answers[key] = ans
    return answers


# ── help / goodbye ────────────────────────────────────────────────────────────


# landing-page command reference: command (purple) · description (white).
_COMMAND_HELP = (
    ("/help", "Show available commands"),
    ("/config", "Update configuration"),
    ("/history", "View previous runs"),
    ("/quit", "Exit CAPO"),
)


def _command_help_table():
    """Aligned two-column table — slash command in purple, description in white.

    A 2-wide leading column gives the same two-space indent as the prose lines
    without Padding (which would expand the grid to the full console width)."""
    from rich.table import Table

    t = Table.grid(padding=(0, 3, 0, 0))
    t.add_column(width=2)                    # indent
    t.add_column(style="cmd", no_wrap=True)  # purple command
    t.add_column(style="default")            # white description
    for cmd, desc in _COMMAND_HELP:
        t.add_row("", cmd, desc)
    return t


def _print_chat_help() -> None:
    console.print()
    console.print("  [brand]CAPO chat[/]  [muted]— tell me what to train, in plain English.[/]\n")
    console.print(command_help_table())  # the full command catalogue
    console.print(
        "\n  [muted]Commands run instantly (no AI). "
        "Just describe your task to start a run.[/]\n"
    )


def _goodbye_chat() -> None:
    console.print()
    console.print("  [brand]Bye 👋[/]  [muted]no run was started.[/]")
    console.print("  [muted]Come back anytime with [/][cmd]capo[/][muted].[/]\n")


# ── deterministic input handling (slash commands never reach the model) ───────


def _handle_command(text: str, cfg: CapoConfig, runs_root: Path) -> tuple[str, Optional[str]]:
    """Classify and act on one input line WITHOUT ever calling the model.

    Returns one of:
      ("quit", None)         leave CAPO
      ("handled", None)      a utility command ran; keep chatting (no model call)
      ("model", instruction) a real message (or a /tune instruction) for the model

    Utility slash commands mirror their `capo <subcommand>` terminal twins
    exactly — /config edits config, /history lists runs, /prune-memory forgets a
    run — and produce NO assistant reply afterwards. The lone exception is
    /tune | /retune, where the user is explicitly asking the model to (re)shape
    the task, so the argument is forwarded as a model turn.
    """
    t = text.strip()
    if not t:
        return ("handled", None)
    if is_quit(t):
        return ("quit", None)
    if not t.startswith("/"):
        return ("model", t)

    parts = t.split(None, 1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    # expand an abbreviated command to its unique full name (/heal → /health);
    # an ambiguous prefix asks the user to narrow it rather than guessing.
    resolved, matches = resolve_slash(cmd, _CHAT_COMMANDS)
    if resolved is not None:
        cmd = resolved
    elif len(matches) >= 2:
        listed = " ".join(f"[cmd]{m}[/]" for m in matches)
        console.print(f"  [muted]{cmd} matches[/] {listed} [muted]— type more to pick one.[/]\n")
        return ("handled", None)

    if cmd == "/quit":
        return ("quit", None)
    if cmd == "/help":
        _print_chat_help()
        return ("handled", None)
    if cmd == "/config":
        from .config import interactive_config_editor, load_config

        interactive_config_editor(load_config(cfg.config_path))
        return ("handled", None)
    if cmd == "/history":
        from .history import print_history

        print_history(runs_root)
        return ("handled", None)
    if cmd == "/prune-memory":
        from .memory import prune_memory

        prune_memory(cfg.repo_root / "runs" / "runs_index.md", run_id=arg or None)
        return ("handled", None)
    if cmd in ("/tune", "/retune"):
        if not arg:
            console.print(f"  [muted]Usage: [/][cmd]{cmd}[/] [cmd.arg]<instruction>[/]\n")
            return ("handled", None)
        return ("model", arg)

    console.print(f"  [muted]Unknown command {cmd} — try [/][cmd]/help[/][muted].[/]\n")
    return ("handled", None)


def _next_message(session: PromptSession, cfg: CapoConfig, runs_root: Path) -> Optional[str]:
    """Prompt until the user gives a real message; run any commands inline.

    Slash commands and plain quit/exit are dispatched deterministically here, so
    they never trigger a model call. Returns the message text for the model, or
    None when the user quits.
    """
    while True:
        kind, payload = _handle_command(_read_user(session), cfg, runs_root)
        if kind == "quit":
            return None
        if kind == "model":
            return payload
        # "handled" → a utility command ran; ask for the next line.


# === finalization ===


def _finalize(cfg: CapoConfig, runs_root: Path, draft: dict, fallback_objective: str) -> ChatPlan:
    """Fill nulls from config defaults and build the structured task.md text."""
    from .run_planner import build_task_markdown

    objective = (draft.get("objective") or fallback_objective or "").strip()
    mode = (draft.get("mode") or "fine-tune").strip()
    dataset_ref = draft.get("dataset_ref") or cfg.dataset_ref
    model_id = draft.get("model_id") or cfg.model_id
    strategy = draft.get("fine_tune_strategy") or cfg.fine_tune_strategy
    gpu = draft.get("gpu_preference") or cfg.gpu_preference
    notes = (draft.get("notes") or "").strip()
    try:
        budget = float(draft.get("max_cost_usd")) if draft.get("max_cost_usd") else cfg.max_cost_usd
    except (TypeError, ValueError):
        budget = cfg.max_cost_usd

    def _opt(key: str) -> Optional[str]:
        v = draft.get(key)
        return v.strip() if isinstance(v, str) and v.strip() else None

    # a structured, scientific brief becomes task.md (free text the agents read).
    enriched, _intent = build_task_markdown(
        objective=objective,
        mode=mode,
        dataset_ref=dataset_ref,
        fine_tune_strategy=strategy,
        max_cost_usd=budget,
        gpu_preference=gpu,
        model_id=model_id,
        runs_root=runs_root,
        title=_opt("title"),
        organism=_opt("organism"),
        target=_opt("target"),
        evaluation=_opt("evaluation"),
        deliverables=_opt("deliverables"),
        constraints=_opt("constraints"),
        notes=notes or None,
    )
    return ChatPlan(
        objective=objective,
        mode=mode,
        dataset_ref=dataset_ref,
        model_id=model_id,
        fine_tune_strategy=strategy,
        gpu_preference=gpu,
        max_cost_usd=budget,
        notes=notes,
        enriched_description=enriched,
    )


# === shared conversation loop ===


def _run_conversation(
    runner,
    session: PromptSession,
    cfg: CapoConfig,
    runs_root: Path,
    draft: dict,
    messages: list[tuple[str, str]],
    first: str,
    *,
    context_prefix: str = "",
    goodbye=_goodbye_chat,
    max_turns: int = 2,
) -> Optional[ChatPlan]:
    """The chat turn-loop shared by the front door and the post-run chat.

    Each turn: build the prompt, take one model turn, parse the strict JSON, then
    reply / ask a question / or finalize a ChatPlan when ready. context_prefix is
    prepended to every model prompt (empty for the front door, so its behaviour is
    byte-identical; the run-results block for the post-run chat). goodbye is the
    farewell printed when the user quits; max_turns lets the post-run agent read
    files before answering. Returns a ChatPlan to launch, or None if the user quit.
    """
    while True:
        prompt = context_prefix + _build_prompt(cfg, runs_root, draft, messages)
        with _Thinking():
            raw = _call_model(runner, prompt, max_turns=max_turns)
        data = _extract_json(raw)

        # a reply with no parseable JSON is a formatting hiccup, not a failure of
        # the user's intent -> so retry once with a strict reminder before giving
        # up. a clear request (e.g. "fine-tune ESM2 on ACE2 binding") then still
        # reaches ready=true and launches, instead of being told "I didn't follow".
        if data is None:
            with _Thinking():
                raw = _call_model(runner, prompt + _JSON_NUDGE, max_turns=max_turns)
            data = _extract_json(raw)

        if data is None:
            # still unstructured. salvage clean prose as a normal reply and keep
            # going. only truly empty / malformed output asks for a rephrase.
            salvaged = _salvage_reply(raw)
            if salvaged:
                data = {"reply": salvaged}
            else:
                console.print(
                    "  [muted]I didn't quite catch that — could you say it another way?[/]\n"
                )
                user = _next_message(session, cfg, runs_root)
                if user is None:
                    goodbye()
                    return None
                messages.append(("user", user))
                continue

        reply = (data.get("reply") or "").strip()
        if reply:
            console.print(f"  {reply}\n")
        _merge_draft(draft, data.get("task") or {})
        messages.append(("assistant", reply or "(thinking)"))

        if data.get("ready"):
            return _finalize(cfg, runs_root, draft, fallback_objective=first)

        questions = data.get("questions") or []
        if questions:
            answers = _ask_questions(questions)
            if answers:
                # answers to known task fields populate the draft directly; the
                # rest become notes. either way they're echoed back to the model.
                notes_bits = []
                for k, v in answers.items():
                    if k in _TASK_KEYS:
                        draft[k] = v
                    else:
                        notes_bits.append(f"{k}={v}")
                if notes_bits:
                    draft["notes"] = "; ".join(
                        p for p in (draft.get("notes"), "; ".join(notes_bits)) if p
                    )
                messages.append(
                    ("user", "Answers: " + "; ".join(f"{k}={v}" for k, v in answers.items()))
                )
            continue

        # just conversing. wait for the next real user message (commands handled
        # inline by _next_message, so they never produce a model reply).
        user = _next_message(session, cfg, runs_root)
        if user is None:
            goodbye()
            return None
        messages.append(("user", user))


# === main loop ===


def run_chat(cfg: CapoConfig, runs_root: Path, initial_intent: Optional[str] = None,
             *, show_welcome: bool = True) -> Optional[ChatPlan]:
    """Hold the conversation; return a ChatPlan to launch, or None if the user quit.

    show_welcome=False suppresses the opening 'Fully ready / what would you like to
    train today' banner + command list — used when re-entering after an abort, where
    that banner is redundant with the caller's own one-line context."""
    from .intent_prompt import parse_intent

    runner = _make_runner(cfg)
    session = _make_session()
    draft: dict = {k: None for k in _TASK_KEYS}
    messages: list[tuple[str, str]] = []

    if show_welcome:
        console.print("  [brand]Fully ready.[/]")
        console.print("  [default]What would you like to train today?[/]\n")
        console.print(_command_help_table())
        console.print()

    # resolve the first real user message. commands (and quit/exit) are handled
    # deterministically here too, none of them ever reach the model.
    if initial_intent is not None:
        kind, payload = _handle_command(initial_intent, cfg, runs_root)
        if kind == "quit":
            _goodbye_chat()
            return None
        first = payload if kind == "model" else _next_message(session, cfg, runs_root)
    else:
        first = _next_message(session, cfg, runs_root)
    if first is None:
        _goodbye_chat()
        return None

    # seed the draft from cheap regex hints so the model starts informed.
    hints = parse_intent(first)
    for k, v in (("dataset_ref", hints.dataset_ref), ("fine_tune_strategy", hints.fine_tune_strategy),
                 ("gpu_preference", hints.gpu_preference), ("max_cost_usd", hints.max_cost_usd)):
        if v:
            draft[k] = v
    messages.append(("user", first))

    return _run_conversation(runner, session, cfg, runs_root, draft, messages, first)


# === post-pipeline chat ===
# After a run finishes, the user lands here: the SAME chat agent, given read-only
# file tools scoped to the run dir so it can answer questions about the results,
# and the SAME launch path (a ChatPlan) so "fine-tune again" routes back into the
# first pipeline rather than inventing a manual flow.

# read-only tools the post-run agent uses to inspect the finished run dir.
_POST_RUN_FILE_TOOLS = ["Read", "Grep", "Glob"]

# the things the user can do, shown once when the post-run chat opens.
_POST_RUN_MENU = (
    "explain the results",
    "inspect metrics",
    "run inference",
    "compare checkpoints",
    "open reports",
    "launch another fine-tuning run",
)


def _print_post_run_intro(run_id: str) -> None:
    console.print()
    console.print("  [brand]Run complete.[/]\n")
    console.print("  [default]You can now ask:[/]")
    for item in _POST_RUN_MENU:
        console.print(f"    [muted]-[/] {item}")
    console.print(
        f"\n  [muted]I can read this run's files to answer. "
        f"Type [/][cmd]/quit[/][muted] when you're done "
        f"(inspect later with [/][cmd]capo inspect {run_id}[/][muted]).[/]\n"
    )


def _goodbye_post_run(run_id: str) -> None:
    console.print()
    console.print(
        f"  [brand]Done 👋[/]  [muted]your run is saved — revisit it with "
        f"[/][cmd]capo inspect {run_id}[/][muted].[/]\n"
    )


def _post_run_context(run_id: str, run_dir: Path, result=None) -> str:
    """A compact run-results block prepended to every post-run model prompt.

    It names the finished run, its terminal state + key metrics (read once from
    final_summary.json), and which artifacts exist — the agent reads the actual
    files via its tools for anything beyond this summary."""
    from .history import _INSPECT_ARTIFACTS  # reuse the canonical artifact list

    lines = [f"finished run: {run_id}", f"run directory (your working dir): {run_dir}"]

    summary = run_dir / "reports" / "final_summary.json"
    if summary.exists():
        try:
            d = json.loads(summary.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            d = {}
        if d.get("terminal_state"):
            lines.append(f"terminal_state: {d['terminal_state']}")
        metrics = d.get("final_metrics") or {}
        if isinstance(metrics, dict) and metrics:
            shown = ", ".join(f"{k}={v}" for k, v in list(metrics.items())[:8])
            lines.append(f"final_metrics: {shown}")
        for key in ("final_model_path", "actual_cost_usd", "trackio_url"):
            if d.get(key) is not None:
                lines.append(f"{key}: {d[key]}")
    elif result is not None and getattr(result, "state", None):
        lines.append(f"terminal_state: {result.state}")

    present = [rel for rel, _lbl in _INSPECT_ARTIFACTS if (run_dir / rel).exists()]
    if present:
        lines.append("artifacts present (read with your file tools): " + ", ".join(present))

    body = "\n  ".join(lines)
    return (
        "RUN RESULTS (this run just finished; your cwd is its directory — read the "
        f"files below for any detail you state):\n  {body}\n\n"
    )


def run_post_run_chat(
    cfg: CapoConfig,
    runs_root: Path,
    run_id: str,
    run_dir: Path,
    result=None,
) -> Optional[ChatPlan]:
    """Hold the post-run conversation; return a ChatPlan to launch another run
    through the same pipeline, or None when the user is done.

    Reuses the front-door machinery end to end — the same JSON contract, the same
    _run_conversation loop, and the same _finalize → ChatPlan that app.py launches
    via _build_orchestrator. The only differences are the post_run_chat system
    prompt, read-only file tools scoped to the run dir, and a results context
    block, so a "fine-tune again" lands in the SAME first pipeline."""
    runner = _make_runner(
        cfg,
        prompt_name="cli/post_run_chat",
        allowed_tools=_POST_RUN_FILE_TOOLS,
        cwd=str(run_dir),
        max_turns=12,
    )
    session = _make_session()
    draft: dict = {k: None for k in _TASK_KEYS}
    messages: list[tuple[str, str]] = []

    _print_post_run_intro(run_id)

    # first real message (slash commands / quit handled inline, never reach model).
    first = _next_message(session, cfg, runs_root)
    if first is None:
        _goodbye_post_run(run_id)
        return None
    messages.append(("user", first))

    context_prefix = _post_run_context(run_id, run_dir, result)
    return _run_conversation(
        runner, session, cfg, runs_root, draft, messages, first,
        context_prefix=context_prefix,
        goodbye=lambda: _goodbye_post_run(run_id),
        max_turns=12,
    )

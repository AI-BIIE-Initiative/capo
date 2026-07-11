"""
Mid-run interactive console.

Architecture (wired in app.py):
  main thread     → orchestrator.run_sync()  (blocks)
  daemon thread A → stream_log()   — Rich prints log lines via patch_stdout
  daemon thread B → run_console()  — prompt_toolkit session + patch_stdout

patch_stdout() is entered here so Rich output from stream_log lands cleanly
above the prompt, and a small refresh_interval keeps those background log writes
flushing live (no keystroke needed). The input is a minimal, transparent command
bar: a dim "›" prompt with a placeholder listing the commands — no heavy bottom
bar, nothing that looks like a stuck text editor. A custom lexer turns a slash
command purple once it resolves to a single command — an exact name or a unique
prefix (so /heal turns purple because it can only mean /health, while an
ambiguous /he stays white). Pressing Enter on a unique prefix runs that command;
an ambiguous prefix prints the candidates and waits for a narrowing keystroke.
Slash commands:
  /help /health /status /tune /config /history /abort /quit
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, ContextManager

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style

from .colors import (
    NOISE,
    PURPLE_0,
    console,
)
from .commands import COMMANDS, command_help_table, is_quit, resolve_slash

# all slash commands available mid-run — Tab-completes; rendered purple when
# fully typed. Derived from the shared registry (run + everywhere commands) so
# the run console and /help never drift apart.
SLASH_COMMANDS = [name for name, _desc, scope in COMMANDS if scope in ("run", "both")]

# prompt_toolkit style: cmd is the purple class the lexer assigns to a fully
# typed command. Mirrors the Rich cmd token (PURPLE_0 bold).
_PT_STYLE = Style.from_dict({"cmd": f"{PURPLE_0} bold", "prompt": f"{PURPLE_0} bold"})


class SlashCommandLexer(Lexer):
    """Colour the first token purple iff it resolves to a single slash command —
    an exact name or a unique prefix (/heal → /health). Ambiguous stays white."""

    def lex_document(self, document):
        lines = document.lines

        def get_line(lineno: int):
            line = lines[lineno]
            stripped = line.lstrip()
            if not stripped:
                return [("", line)]
            indent = line[: len(line) - len(stripped)]
            first = stripped.split(" ", 1)[0]
            rest = stripped[len(first):]
            if resolve_slash(first, SLASH_COMMANDS)[0] is not None:
                frags = []
                if indent:
                    frags.append(("", indent))
                frags.append(("class:cmd", first))
                if rest:
                    frags.append(("", rest))
                return frags
            return [("", line)]

        return get_line


def read_phase(state_path: Path) -> str:
    """Return current_phase from state.json, or 'init' if unreadable."""
    try:
        return json.loads(state_path.read_text(encoding="utf-8")).get("current_phase", "init")
    except Exception:
        return "init"


def fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# the input line itself is the command bar: a dim "›" prompt + this placeholder.
_PROMPT = HTML(f'<style fg="{NOISE}">  › </style>')
_PLACEHOLDER = HTML(
    f'<style fg="{NOISE}">type a command  ·  /health  /abort  /history  /quit</style>'
)


def elapsed_rprompt(start_time: float) -> HTML:
    """Right-hand clock on the run prompt: just the elapsed running time.

    Re-evaluated on every prompt_toolkit redraw (the session's refresh_interval
    drives those), so it ticks up live. Deliberately minimal — no 'LIVE' label
    and no blinking dot following the cursor; liveness is signalled once by the
    top banner, while this keeps the running time always in view."""
    return HTML(f'<style fg="{NOISE}">{fmt_elapsed(time.monotonic() - start_time)} </style>')


# the live banner: a small dot + lowercase "live", shown the instant a run starts.
_BANNER_SUFFIX = "[brand]live[/]  [muted]· streaming run logs below · type a command anytime[/]"
_BANNER_ON = f"  [phase.run]●[/] {_BANNER_SUFFIX}"


def print_run_banner(run_id: str) -> None:
    """Print the live-run banner exactly once, statically, above the streamed logs.
    Used only by the plain (non-full-screen) run path. 
    the full-screen RunConsole renders its own top bar."""
    console.print(_BANNER_ON + "\n")


def save_note(run_dir: Path, text: str) -> None:
    """Append a timestamped note to user_notes.txt in the run dir."""
    notes = run_dir / "user_notes.txt"
    ts = datetime.now().strftime("%H:%M:%S")
    notes.parent.mkdir(parents=True, exist_ok=True)
    with notes.open("a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {text}\n")
    console.print("  [ok]✓[/] Saved  [muted]→ user_notes.txt[/]\n")


def _print_help() -> None:
    console.print()
    console.print("  [brand]Commands[/]  [muted](Tab-completes; purple when fully typed)[/]\n")
    console.print(command_help_table("run"))  # the full run-time command catalogue
    console.print(
        "\n  [muted]/quit detaches this console (the run keeps going); "
        "plain text is saved as a run note.[/]\n"
    )


# === pending-question surfacing ===
# the gate writes reports/pending_question.json and pauses by EXITING; resume via
# capo resume. these helpers only READ that artifact and render it, the
# pause/resume contract (who writes it, who applies the answer) is unchanged.


def read_pending_question(run_dir: Path, *, state: dict | None = None) -> dict | None:
    """Load a paused run's pending question, or None if there isn't one.

    Prefers state.json's pending_question_path (the gate records it there),
    falling back to the conventional reports/pending_question.json. Any read /
    parse failure yields None — surfacing the question is best-effort and never
    blocks the summary."""
    rel = (state or {}).get("pending_question_path") or "reports/pending_question.json"
    path = run_dir / rel
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def print_pending_question(question: dict) -> None:
    """Render a paused run's question + options prominently, so the decision
    that unblocks the run is impossible to miss. Presentation only — the
    question already lives on disk and answering still flows through resume."""
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    header = str(question.get("header") or "").strip()
    qtext = str(question.get("question") or "").strip()
    options = question.get("options") or []

    body = Table.grid(padding=(0, 1))
    body.add_column()
    if header:
        body.add_row(Text(header, style="brand"))
    if qtext:
        body.add_row(Text(qtext, style="brand.dim"))
    if options:
        body.add_row(Text(""))
        body.add_row(Text("Options", style="metric.key"))
        opts = Table.grid(padding=(0, 1))
        opts.add_column(style="cmd", no_wrap=True)    # 1-based index
        opts.add_column(style="brand", no_wrap=True)  # label
        opts.add_column(style="muted")                # description
        for i, opt in enumerate(options, 1):
            lbl = str(opt.get("label") or "").strip()
            desc = str(opt.get("description") or "").strip()
            opts.add_row(str(i), lbl, f"— {desc}" if desc else "")
        body.add_row(opts)

    console.print()
    console.print(
        Panel(
            body,
            title="[brand] Action needed[/]",
            subtitle="[muted]answer below, or later with capo resume[/]",
            border_style="brand.dim",
            padding=(0, 1),
        )
    )


def _is_free_text_question(question: dict) -> bool:
    """A question wants free text (not a pick) when its answer is appended as
    label semantics, or its only option is the generic 'Provide details'."""
    if question.get("answer_target") == "profile.label_semantics":
        return True
    options = question.get("options") or []
    labels = [str(o.get("label") or "").strip() for o in options]
    return len(labels) == 1 and labels[0].lower() == "provide details"


def prompt_pending_answer(question: dict) -> str | None:
    """Collect an answer to a paused run's pending question, inline.

    Returns the answer string to forward to resume_run(answer=...), or None if
    the user defers (Enter / later / Ctrl-C / EOF → answer later via
    capo resume). This only COLLECTS the answer; resume_run still applies it
    and re-enters the gate, so the pause/resume contract is untouched."""
    from rich.prompt import Prompt

    options = question.get("options") or []
    labels = [str(o.get("label") or "").strip() for o in options if o.get("label")]

    try:
        if _is_free_text_question(question):
            ans = Prompt.ask(
                "  [brand]Your answer[/] [muted](or Enter to answer later)[/]",
                default="", show_default=False, console=console,
            ).strip()
            return ans or None
        raw = Prompt.ask(
            "  [brand]Your choice[/] [muted](number, label, or Enter to answer later)[/]",
            default="", show_default=False, console=console,
        ).strip()
        if not raw or raw.lower() in {"later", "l"}:
            return None
        if raw.isdigit():
            idx = int(raw) - 1
            return labels[idx] if 0 <= idx < len(labels) else None
        for lbl in labels:  # match a label case-insensitively
            if lbl.lower() == raw.lower():
                return lbl
        return raw  # free-form passthrough — resume_run validates per answer_target
    except (EOFError, KeyboardInterrupt):
        return None


def print_run_summary(
    run_id: str,
    run_dir: Path,
    *,
    result=None,
    error: Exception | None = None,
    paused: bool = False,
    pause_reason: str | None = None,
    pending_question: dict | None = None,
) -> None:
    """Render the end-of-run summary inside the same brand-dim purple box used
    for the pre-launch 'Ready to launch' panel — so a finished run closes as
    cleanly as it opened. Carries the exact same facts the plain end-of-run
    prints did (state, model, trackio, pause, run dir, health/resume hints),
    just framed. Presentation only; nothing about the run changes."""
    from rich.panel import Panel
    from rich.table import Table

    # status line + panel title depend on the terminal state.
    if error is not None:
        status = f"[err]✗ failed[/]  [muted]{type(error).__name__}: {error}[/]"
        title = "[err] Run failed[/]"
    elif result is not None and getattr(result, "state", None) == "completed":
        status, title = "[ok]✓ completed[/]", "[brand] Run complete[/]"
    elif result is not None:
        status = f"[err]✗ {result.state}[/]"
        title = f"[brand] Run {result.state}[/]"
    else:
        status, title = "[muted]— no result[/]", "[brand] Run finished[/]"

    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column("Key", style="metric.key", no_wrap=True)
    t.add_column("Value", style="brand.dim")
    t.add_row("Status", status)
    t.add_row("Run", run_id)
    if result is not None and getattr(result, "finetuned_model_path", None):
        t.add_row("Model", str(result.finetuned_model_path))
    if result is not None and getattr(result, "trackio_url", None):
        t.add_row("Trackio", str(result.trackio_url))
    if paused:
        t.add_row("Paused", f"[metric.warn]⏸ paused[/]  [muted]{pause_reason or 'awaiting input'}[/]")
    t.add_row("Run dir", str(run_dir))
    t.add_row("Health", f"[cmd]capo health {run_id}[/]")
    t.add_row("Resume", f"[cmd]capo resume {run_id}[/]")

    console.print()
    console.print(Panel(t, title=title, border_style="brand.dim", padding=(0, 1)))
    console.print()

    # when paused for user input, show the actual question + options right below
    # the summary so the decision is impossible to miss (#3). still advisory of
    # capo resume; the inline-answer prompt (if any) is wired by the caller.
    if paused and pending_question:
        print_pending_question(pending_question)
        console.print()


# ── shared interaction router ─────────────────────────────────────────────────
# ctx.capture() wraps each command's console.print(...) calls so the full-screen owner can grab
# the rendered ANSI and show it inside its log pane, while the plain path prints
# straight through (a no-op nullcontext).


@dataclass
class RunContext:
    """Everything a run command needs, plus the seam that decides where its
    rendered output goes (direct console vs. captured into a log pane)."""

    run_id: str
    run_dir: Path
    runs_root: Path
    stop_event: threading.Event
    start_time: float
    # set by /abort: a decisive "the user asked to stop" signal that the launcher
    # reads after run_sync returns. distinct from stop_event (which is also set on
    # normal completion) and from the SIGINT (which the orchestrator may swallow),
    # so an abort is never mistaken for a finished run.
    abort_event: threading.Event = field(default_factory=threading.Event)
    # only a confirmed abort sets abort_event + tears the run + GPU down. mutable single
    # field (not an Event) because it is flipped on and off, only ever on this thread.
    abort_pending: bool = False
    # context-manager factory wrapping a block of console.print(...) calls.
    capture: Callable[[], ContextManager] = field(default=lambda: nullcontext())
    # whether commands that spawn their own prompt_toolkit UI (the /config editor,
    # the interactive /prune-memory picker) may run. False under the full-screen
    # owner, where a nested full-screen app would fight the active one.
    allow_nested_ui: bool = True


def _detach_notice(run_id: str, run_dir: Path) -> None:
    console.print(
        f"\n  [brand]Detaching console[/]  [muted]— the run keeps going.[/]\n"
        f"  [muted]run dir:  [/]{run_dir}\n"
        f"  [muted]check it: [/][cmd]capo health {run_id}[/]"
        f"   [muted]stop it: [/][cmd]/abort[/][muted] or Ctrl+C[/]\n"
    )


def _print_status(ctx: RunContext) -> None:
    phase = read_phase(ctx.run_dir / "state.json")
    style = (
        "phase.done" if phase == "completed"
        else "phase.fail" if phase == "failed"
        else "phase.run"
    )
    console.print(
        f"\n  [{style}]{phase}[/]  "
        f"[muted]{fmt_elapsed(time.monotonic() - ctx.start_time)}[/]\n"
    )


# === abort: arm → confirm → tear down ===
# /abort and Ctrl+C share this two-step so both produce the same flow: the first
# press arms a confirmation (a gentle checkpoint-loss warning), and only an
# explicit 'y' (or a second /abort / Ctrl+C) commits it. Until then the run is
# untouched, so a stray interrupt never kills the run + GPU by accident.


def _arm_abort(ctx: RunContext) -> None:
    """Ask the user to confirm a stop; the run keeps going until they answer."""
    with ctx.capture():
        console.print(
            "\n  [accent]Stop this run?[/]  [muted]Any training progress since the "
            "last saved checkpoint will be lost.[/]\n"
            "  [muted]Type [/][cmd]y[/][muted] to confirm — anything else keeps the "
            "run going.[/]\n"
        )
    ctx.abort_pending = True


def _do_abort(ctx: RunContext) -> str:
    """Commit the abort: signal the orchestrator to stop. The launcher (app.py)
    owns the visible teardown — stop training, sync data off, terminate + verify."""
    with ctx.capture():
        console.print(
            "\n  [err]Aborting the run[/]  [muted]— saving your data off the instance "
            "and shutting down the GPU. Hold on…[/]\n"
        )
    ctx.abort_event.set()   # decisive: survives a SIGINT the orchestrator swallows
    ctx.stop_event.set()
    os.kill(os.getpid(), signal.SIGINT)  # interrupt run_sync in the main thread
    return "abort"


def dispatch_run_command(text: str, ctx: RunContext) -> str:
    """Process one input line. Returns the next action for the caller's loop:
    'continue' (keep the console open), 'detach' (leave; run keeps going),
    'abort' (stop confirmed — SIGINT already sent to the orchestrator), or
    'narrow' (an abbreviated command matched 2+ commands; the candidates were
    printed and the caller should keep the line so the user can type more)."""
    text = text.strip()
    if not text:
        return "continue"

    # an armed abort consumes this line as its yes/no answer (set by /abort or a
    # Ctrl+C). only an explicit 'y' (or a repeated /abort) commits — anything else
    # cancels and the run carries on, untouched.
    if ctx.abort_pending:
        ctx.abort_pending = False
        low = text.lower()
        if low in ("y", "yes") or low.startswith("/abort"):
            return _do_abort(ctx)
        with ctx.capture():
            console.print("  [muted]Okay — keeping the run going.[/]\n")
        return "continue"

    if is_quit(text):
        with ctx.capture():
            _detach_notice(ctx.run_id, ctx.run_dir)
        return "detach"

    parts = text.split(None, 1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    # expand an abbreviated slash command to its unique full name (/heal → /health);
    # on an ambiguous prefix, list the candidates and ask for one more keystroke
    # rather than guessing. a non-slash line (a free-text note) is left untouched.
    if cmd.startswith("/"):
        resolved, matches = resolve_slash(cmd, SLASH_COMMANDS)
        if resolved is not None:
            cmd = resolved
        elif len(matches) >= 2:
            with ctx.capture():
                listed = " ".join(f"[cmd]{m}[/]" for m in matches)
                console.print(
                    f"  [muted]{cmd} matches[/] {listed} [muted]— type more to pick one.[/]\n"
                )
            return "narrow"
        # zero matches → fall through; an unknown slash line is saved as a note.

    if cmd == "/abort":
        _arm_abort(ctx)
        return "continue"

    action = "continue"

    with ctx.capture():
        if cmd == "/help":
            _print_help()
        elif cmd == "/health":
            from .health import print_health_card  # lazy: no hard dep on health.py

            print_health_card(ctx.run_id, ctx.runs_root)
        elif cmd == "/status":
            _print_status(ctx)
        elif cmd == "/tune":
            if not args:
                console.print("  [muted]Usage: [/][cmd]/tune[/] [cmd.arg]<message>[/]\n")
            else:
                save_note(ctx.run_dir, args)
        elif cmd == "/config":
            if ctx.allow_nested_ui:
                from .config import interactive_config_editor, load_config

                interactive_config_editor(load_config())
            else:
                console.print(
                    "  [muted]Edit config after the run — [/][cmd]capo config[/]"
                    "[muted] (not available mid-run in this view).[/]\n"
                )
        elif cmd == "/history":
            from .history import print_history

            print_history(ctx.runs_root)
        elif cmd == "/prune-memory":
            if ctx.allow_nested_ui or args:
                from .config import load_config
                from .memory import prune_memory

                index = load_config().repo_root / "runs" / "runs_index.md"
                prune_memory(index, run_id=args or None)
            else:
                console.print(
                    "  [muted]Forget a run after this one — [/][cmd]capo prune-memory[/]"
                    "[muted].[/]\n"
                )
        elif cmd == "/quit":
            _detach_notice(ctx.run_id, ctx.run_dir)
            action = "detach"
        else:
            save_note(ctx.run_dir, text)
            console.print(
                "  [muted]Saved as a run note "
                "([/][cmd]/tune[/][muted] <message> does the same).[/]\n"
            )
    return action


def run_console(
    run_id: str,
    run_dir: Path,
    stop_event: threading.Event,
    runs_root: Path,
    abort_event: threading.Event | None = None,
) -> None:
    """Interactive console. Run in a daemon thread; set stop_event to exit.

    /abort sets abort_event + stop_event and sends SIGINT to the process so the
    orchestrator in the main thread is interrupted (best-effort cooperative stop);
    abort_event lets the launcher tell an abort apart from a normal finish.
    """
    abort_event = abort_event or threading.Event()
    # Disable prompt_toolkit's cursor-position-request (CPR) probe for this view.
    # CPR asks the terminal "where is the cursor?" and waits up to 2s for a reply
    # to position its "draw above the prompt" rendering. Terminals that never
    # answer (some IDE / embedded terminals) leave the background log lines
    # unpainted until the prompt is dismissed — the exact "nothing shows until I
    # press Ctrl+D" symptom. With CPR off, prompt_toolkit assumes a known cursor
    # row and paints streamed logs immediately. Safe here: this is a single-line
    # prompt, not a full-screen app. setdefault so an explicit user value wins.
    os.environ.setdefault("PROMPT_TOOLKIT_NO_CPR", "1")

    # this runs in a daemon thread; a fresh thread has no asyncio event loop in
    # py3.12+, and PromptSession.prompt() needs one. create it before any prompt.
    import asyncio

    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    start_time = time.monotonic()
    ctx = RunContext(
        run_id=run_id,
        run_dir=run_dir,
        runs_root=runs_root,
        stop_event=stop_event,
        start_time=start_time,
        abort_event=abort_event,
    )  # plain path: capture=nullcontext (print straight through), nested UI allowed

    # Belt-and-suspenders with the env var above: disable CPR on the output
    # object itself. responds_to_cpr is False when enable_cpr is False, so
    # run_in_terminal never awaits wait_for_cpr_responses() — the await that, in
    # terminals which claim CPR support but never answer (some IDE terminals),
    # blocks the *first* background write and chains every later log line behind
    # it, so nothing paints until the prompt is torn down (the "logs only after
    # Ctrl+D" bug). The env var only disables CPR if read in time; setting the
    # attribute guarantees it for the app's own output regardless of timing.
    from prompt_toolkit.output.defaults import create_output

    output = create_output()
    if hasattr(output, "enable_cpr"):
        output.enable_cpr = False

    session: PromptSession = PromptSession(
        history=InMemoryHistory(),
        auto_suggest=AutoSuggestFromHistory(),
        completer=WordCompleter(SLASH_COMMANDS, sentence=True),
        complete_while_typing=True,
        lexer=SlashCommandLexer(),
        style=_PT_STYLE,
        output=output,
        # redraw a few times a second so background log writes (streamed via
        # patch_stdout from the other threads) flush live without a keystroke.
        refresh_interval=0.5,
    )

    print_run_banner(run_id)  # signal the live view before the first log arrives

    # minimal running clock on the right edge (re-evaluated each refresh tick) —
    # no "LIVE" label or blinking dot following the cursor; the top banner already
    # signalled liveness.
    def _rprompt() -> HTML:
        return elapsed_rprompt(start_time)

    with patch_stdout():
        while not stop_event.is_set():
            try:
                user_input = session.prompt(_PROMPT, placeholder=_PLACEHOLDER, rprompt=_rprompt)
                if dispatch_run_command(user_input, ctx) in ("detach", "abort"):
                    break
            except KeyboardInterrupt:
                # Ctrl+C routes through the SAME arm → confirm path as /abort, so
                # both behave identically: first press asks, a second (or a typed y)
                # commits.
                if ctx.abort_pending:
                    ctx.abort_pending = False
                    if _do_abort(ctx) == "abort":
                        break
                else:
                    _arm_abort(ctx)
            except EOFError:
                # Ctrl+D detaches the console; the run keeps going in the background.
                console.print(
                    f"\n  [brand]Detaching console[/]  [muted]— the run keeps going.[/]\n"
                    f"  [muted]check it: [/][cmd]capo health {run_id}[/]\n"
                )
                break

"""
Canonical slash-command registry, shared by the chat layer and the run console.

One source of truth so /help lists *every* command (purple name + white
description, the user-requested layout) and so the two surfaces stay in lockstep
with the equivalent capo <subcommand> terminal commands. Commands are
dispatched deterministically by chat.py / run_console.py — this module only
holds the catalogue and renders the help table; it never calls the model.

scope:
  "chat" — only meaningful at the conversational front door
  "run"  — only meaningful while a run is live
  "both" — available everywhere
"""

from __future__ import annotations

from rich.table import Table

# (name, description, scope). Order is the display order in /help.
COMMANDS: tuple[tuple[str, str, str], ...] = (
    ("/help", "Show all commands", "both"),
    ("/config", "Edit configuration", "both"),
    ("/history", "Show previous runs", "both"),
    ("/health", "Show current run status", "run"),
    ("/status", "One-line run phase + elapsed", "run"),
    ("/abort", "Stop the active run", "run"),
    ("/tune", "Refine the task with an instruction", "both"),
    ("/retune", "Modify the current task with an instruction", "chat"),
    ("/prune-memory", "Remove a run from memory by run ID", "both"),
    ("/quit", "Exit CAPO", "both"),
)

# typing quit / exit (no slash) means the same as /quit.
QUIT_WORDS: frozenset[str] = frozenset({"quit", "exit", "/quit", ":q", "q"})

# the set of recognised command names (for "unknown command" detection).
COMMAND_NAMES: frozenset[str] = frozenset(name for name, _d, _s in COMMANDS)


def is_quit(text: str) -> bool:
    """True when the input means 'leave CAPO' (quit / exit / /quit)."""
    return text.strip().lower() in QUIT_WORDS


def resolve_slash(token: str, names) -> tuple[str | None, list[str]]:
    """Resolve a possibly-abbreviated slash token against known command names.

    Returns (resolved, matches):
      - exact match           → (name, [name])            e.g. /health → /health
      - unique prefix          → (full_name, [full_name])  e.g. /heal   → /health
      - ambiguous prefix (2+)  → (None, [c1, c2, ...])     caller asks to narrow
      - no match               → (None, [])                caller treats as unknown

    Case-insensitive on the token. Callers decide what a None resolution means:
    on Enter, expand when resolved is not None; when matches has 2+ entries, ask
    the user to type a bit more; an empty matches list is an unknown command.
    """
    token = token.lower()
    names = list(names)
    if token in names:                       # exact wins even if it prefixes another
        return token, [token]
    matches = [n for n in names if n.startswith(token)]
    if len(matches) == 1:
        return matches[0], matches
    return None, matches                     # 0 or 2+ → no single resolution


def command_help_table(scope: str | None = None) -> Table:
    """Aligned two-column help — slash command in purple, description in white.

    scope=None lists every command; scope="chat"/"run" filters to the commands
    that apply there (plus the "both" commands). A 2-wide leading column gives
    the standard two-space indent without Padding (which would stretch the grid
    to the full console width).
    """
    t = Table.grid(padding=(0, 3, 0, 0))
    t.add_column(width=2)                     # indent
    t.add_column(style="cmd", no_wrap=True)   # purple command
    t.add_column(style="default")             # white description
    for name, desc, sc in COMMANDS:
        if scope and sc != "both" and sc != scope:
            continue
        t.add_row("", name, desc)
    return t

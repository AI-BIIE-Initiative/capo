"""
Small inline prompt_toolkit widgets: an arrow-key single-select (with a free-text
"Other…" escape hatch) and a styled text input.

These are the building blocks for the chat layer's clarifying questions and the
interactive config editor. The selector renders inline (not full-screen) and
erases itself when done, so it composes cleanly above the Rich-rendered console.

_select is the single seam the TUI runs through; tests monkeypatch it to drive
a choice without a real terminal.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style as PTStyle

from .colors import NOISE, PURPLE_0, console

OTHER = "Other…"

_SELECT_STYLE = PTStyle.from_dict({"sel": f"{PURPLE_0} bold", "opt": "", "hint": NOISE})


def _ensure_event_loop() -> None:
    """A fresh daemon thread (e.g. the run console) has no asyncio loop; the
    prompt_toolkit Application needs one. Create it lazily, once."""
    import asyncio

    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


def _select(lines_fn: Callable[[int], list], count: int, default_idx: int = 0) -> Optional[int]:
    """Core arrow-key loop. Returns the chosen index, or None if cancelled.

    lines_fn(current_idx) returns prompt_toolkit (style, text) fragments for the
    whole list given which row is highlighted.
    """
    _ensure_event_loop()
    state = {"idx": max(0, min(default_idx, count - 1))}
    kb = KeyBindings()

    @kb.add("up")
    @kb.add("c-p")
    @kb.add("k")
    def _up(_e) -> None:
        state["idx"] = (state["idx"] - 1) % count

    @kb.add("down")
    @kb.add("c-n")
    @kb.add("j")
    def _down(_e) -> None:
        state["idx"] = (state["idx"] + 1) % count

    @kb.add("enter")
    def _ok(e) -> None:
        e.app.exit(result=state["idx"])

    @kb.add("c-c")
    @kb.add("escape")
    def _cancel(e) -> None:
        e.app.exit(result=None)

    control = FormattedTextControl(lambda: lines_fn(state["idx"]), focusable=True, show_cursor=False)
    app: Application = Application(
        layout=Layout(HSplit([Window(control, height=count)])),
        key_bindings=kb,
        style=_SELECT_STYLE,
        full_screen=False,
        erase_when_done=True,
        mouse_support=False,
    )
    return app.run()


def _opt_lines(options: Sequence[str], cur: int) -> list:
    frags: list = []
    for i, opt in enumerate(options):
        pointer = "❯ " if i == cur else "  "
        cls = "class:sel" if i == cur else "class:opt"
        frags.append((cls, f"   {pointer}{opt}\n"))
    return frags


def text_input(label: str, default: Optional[str] = None) -> str:
    """One-line styled free-text input; empty submit keeps the default."""
    _ensure_event_loop()
    try:
        val = pt_prompt(
            HTML(f'<style fg="{PURPLE_0}"><b>   {label} ❯ </b></style>'),
            default=default or "",
        )
    except (KeyboardInterrupt, EOFError):
        return default or ""
    val = val.strip()
    return val or (default or "")


def select_one(
    label: str,
    choices: Sequence[str],
    default: Optional[str] = None,
    allow_other: bool = True,
    other_prompt: str = "Type your answer",
) -> str:
    """Inline arrow-key single-select. Appends an 'Other…' free-text option.

    Returns the chosen string (or the typed text when 'Other…' is picked).
    """
    options = list(choices)
    if allow_other:
        options.append(OTHER)
    default_idx = options.index(default) if default in options else 0

    console.print(f"  [prompt.label]{label}[/]  [prompt.hint](↑/↓ · Enter)[/]")
    chosen = _select(lambda cur: _opt_lines(options, cur), len(options), default_idx)
    if chosen is None:
        raise KeyboardInterrupt
    sel = options[chosen]
    if allow_other and sel == OTHER:
        sel = text_input(other_prompt, default=None)
    console.print(f"  [ok]✓[/] [brand.dim]{sel}[/]\n")
    return sel


def select_index(
    label: str,
    rows: Sequence[str],
    default_idx: int = 0,
    hint: str = "↑/↓ · Enter to edit · Esc to finish",
) -> Optional[int]:
    """Menu selector returning the chosen row index (None on Esc).

    rows are pre-formatted plain strings (the caller styles them); used by the
    config editor where each row is a 'field   value' line.
    """
    console.print(f"  [prompt.label]{label}[/]  [prompt.hint]({hint})[/]")
    return _select(lambda cur: _opt_lines(rows, cur), len(rows), default_idx)

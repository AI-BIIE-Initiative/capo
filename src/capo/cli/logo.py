"""ANSI-Shadow CAPO wordmark with a per-character colour gradient + info box.

Ported from the reference logo.tsx: the wordmark is coloured column-by-column
with a two-segment gradient (blue → muted-purple → pink), so the letters fade
horizontally. The box below carries the subtitle and version in a dim tone.
"""

from __future__ import annotations

from rich.align import Align
from rich.console import Group
from rich.text import Text

from .colors import gradient_hex, console

# ANSI Shadow font — "CAPO", 35 chars wide × 6 rows (matches logo.ts).
_LINES = [
    " ██████╗  █████╗ ██████╗   ██████╗ ",
    "██╔════╝ ██╔══██╗██╔══██╗ ██╔═══██╗",
    "██║      ███████║██████╔╝ ██║   ██║",
    "██║      ██╔══██║██╔═══╝  ██║   ██║",
    "╚██████╗ ██║  ██║██║      ╚██████╔╝",
    " ╚═════╝ ╚═╝  ╚═╝╚═╝       ╚═════╝ ",
]

_SUBTITLE = "Compute-Aware Automated Protein Optimization"


def _version() -> str:
    # read the installed package version; fall back to the pyproject value.
    try:
        from importlib.metadata import version as _pkg_version

        return "v" + _pkg_version("capo")
    except Exception:
        return "v0.1.0"


def _gradient_rows() -> list[Text]:
    """Each art row, coloured per-character by column position."""
    width = max(len(ln) for ln in _LINES)
    rows: list[Text] = []
    for line in _LINES:
        t = Text()
        for i, ch in enumerate(line):
            if ch == " ":
                t.append(" ")
            else:
                t.append(ch, style=gradient_hex(i / (width - 1)) + " bold")
        rows.append(t)
    return rows


def _box() -> list[Text]:
    """A 50-wide rule box: title · subtitle · version (dim)."""
    title = " CAPO CLI "
    side = (50 - 2 - len(title)) // 2  # ╭ + ─… title …─ + ╮
    top = "╭" + "─" * side + title + "─" * (50 - 2 - side - len(title)) + "╮"
    bottom = "╰" + "─" * 48 + "╯"

    def _center(text: str) -> str:
        pad = 48 - len(text)
        left = pad // 2
        return "│" + " " * left + text + " " * (pad - left) + "│"

    style = "brand.dim"
    return [
        Text(top, style=style),
        Text(_center(_SUBTITLE), style=style),
        Text(_center(_version()), style=style),
        Text(bottom, style=style),
    ]


def print_logo() -> None:
    """Print the centred gradient CAPO wordmark and info box."""
    console.print()
    console.print(Align.center(Group(*_gradient_rows())))
    console.print(Align.center(Group(*_box())))
    console.print()

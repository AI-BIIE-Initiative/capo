"""
CLI colour tokens derived from src/capo/viz/palette.py.

All Rich markup uses these semantic names, never raw hex strings. Slash
commands always render via the cmd token (PURPLE_0, bold) so a fully typed
command (e.g. /health) shows purple while partial/unknown text stays default.
"""

from __future__ import annotations

from rich.console import Console
from rich.style import Style
from rich.theme import Theme

# hex values — mirror palette.py exactly
_PURPLE_0 = "#713D8F"
_PURPLE_50 = "#C694E1"
_ORANGE_0 = "#9B3208"
_ORANGE_50 = "#E6905B"
_BLUE_0 = "#1E5994"
_BLUE_50 = "#8DB8E2"
_GREEN_0 = "#0E625C"
_GREEN_50 = "#78B5B0"
_NOISE = "#AAAAAA"

THEME = Theme(
    {
        # --- brand ---
        "brand": Style(color=_PURPLE_0, bold=True),
        "brand.dim": Style(color=_PURPLE_50),
        # --- phase / progress ---
        "phase.done": Style(color=_GREEN_0, bold=True),
        "phase.run": Style(color=_ORANGE_50, bold=True),
        "phase.wait": Style(color=_NOISE),
        "phase.fail": Style(color=_ORANGE_0, bold=True),
        # --- log line tags (see progress.py prefix conventions) ---
        "tag.lambda": Style(color=_BLUE_0, bold=True),
        "tag.tmux": Style(color=_BLUE_50),
        "tag.rsync": Style(color=_PURPLE_50),
        "tag.setup": Style(color=_GREEN_50),
        "tag.training": Style(color=_GREEN_0, bold=True),
        "tag.agent": Style(color=_ORANGE_50),
        "tag.shell": Style(color=_NOISE),
        "tag.hardware": Style(color=_BLUE_50),
        "tag.results": Style(color=_GREEN_0),
        "tag.summary": Style(color=_PURPLE_0, bold=True),
        "tag.default": Style(color=_NOISE),
        # --- log SOURCES (the runner that produced a line; one colour each) ---
        # the orchestrator itself stays white (no colour) — see log_streamer.
        "src.orchestrator": Style(bold=True),               # white / default, bold
        "src.research": Style(color=_BLUE_0, bold=True),     # HF researcher
        "src.memory": Style(color=_PURPLE_0, bold=True),     # episodic memory
        "src.data": Style(color=_GREEN_0, bold=True),        # data-profiler agent
        "src.infra": Style(color=_BLUE_50, bold=True),       # infrastructure agent
        "src.model": Style(color=_PURPLE_50, bold=True),     # model-selection agent
        "src.health": Style(color=_GREEN_50, bold=True),     # training-health-monitor
        "src.tracker": Style(color=_PURPLE_50, bold=True),   # experiment-tracker
        # --- log SEMANTICS (override colour by meaning) ---
        "log.ts": Style(color=_NOISE, dim=True),             # timestamp, very dim
        "log.warn": Style(color=_ORANGE_50, bold=True),
        "log.err": Style(color=_ORANGE_0, bold=True),
        "log.ok": Style(color=_GREEN_0, bold=True),
        "log.progress": Style(color=_GREEN_50),
        "log.event": Style(color=_NOISE),                    # inner [event] tag, dim
        "log.cont": Style(color=_NOISE, italic=True),        # folded continuation line
        "log.noise": Style(color=_NOISE, dim=True),          # cache/compaction/*.cmd
        # --- health card ---
        "metric.good": Style(color=_GREEN_0, bold=True),
        "metric.warn": Style(color=_ORANGE_50, bold=True),
        "metric.bad": Style(color=_ORANGE_0, bold=True),
        "metric.key": Style(color=_BLUE_50),
        # --- table ---
        "table.header": Style(color=_PURPLE_0, bold=True),
        "table.id": Style(color=_BLUE_50),
        "table.done": Style(color=_GREEN_0),
        "table.fail": Style(color=_ORANGE_0),
        "table.run": Style(color=_ORANGE_50, bold=True),
        # --- prompts ---
        "prompt.label": Style(color=_PURPLE_50, bold=True),
        "prompt.hint": Style(color=_NOISE, italic=True),
        # --- slash commands — always PURPLE_0, always bold ---
        "cmd": Style(color=_PURPLE_0, bold=True),
        "cmd.arg": Style(color=_PURPLE_50),
        # --- generic ---
        "muted": Style(color=_NOISE, italic=True),
        "accent": Style(color=_ORANGE_50),
        "ok": Style(color=_GREEN_0, bold=True),
        "err": Style(color=_ORANGE_0, bold=True),
    }
)

# shared console — import this everywhere; never create an ad-hoc Console().
# highlight=False keeps Rich from auto-recolouring numbers/paths in log lines.
console = Console(theme=THEME, highlight=False)

# raw hex re-exported for prompt_toolkit HTML (which cannot use Rich theme names)
PURPLE_0 = _PURPLE_0
PURPLE_50 = _PURPLE_50
ORANGE_0 = _ORANGE_0
ORANGE_50 = _ORANGE_50
GREEN_0 = _GREEN_0
GREEN_50 = _GREEN_50
BLUE_0 = _BLUE_0
BLUE_50 = _BLUE_50
NOISE = _NOISE

# logo gradient stops (blue → muted-purple → pink). Distinct from the plotting
# palette on purpose — these are the brand wordmark colours, not chart colours.
LOGO_C0 = "#5B6FA6"
LOGO_C1 = "#8A7FB5"
LOGO_C2 = "#C792B6"


def gradient_hex(t: float) -> str:
    """Two-segment linear gradient LOGO_C0→LOGO_C1→LOGO_C2 at position t∈[0,1]."""

    def _lerp(a: str, b: str, s: float) -> tuple[int, int, int]:
        ar, ag, ab = int(a[1:3], 16), int(a[3:5], 16), int(a[5:7], 16)
        br, bg, bb = int(b[1:3], 16), int(b[3:5], 16), int(b[5:7], 16)
        return (
            round(ar + (br - ar) * s),
            round(ag + (bg - ag) * s),
            round(ab + (bb - ab) * s),
        )

    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    if t <= 0.5:
        r, g, b = _lerp(LOGO_C0, LOGO_C1, t * 2)
    else:
        r, g, b = _lerp(LOGO_C1, LOGO_C2, (t - 0.5) * 2)
    return f"#{r:02X}{g:02X}{b:02X}"

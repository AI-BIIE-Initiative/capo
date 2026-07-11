"""Scientific markdown tables + value formatting.

Pure functions, no I/O. The finalizer authors RUN_REPORT.md and the cost /
recovery sections through these helpers so every run formats numbers the same
way: 4 decimals for metrics and loss, `$X.XX` for cost, human-readable
durations, and an em dash (—) for anything missing.

Design notes:
- markdown_table renders whatever the caller puts in each cell. Pre-format
  values with the fmt_* helpers when you want a specific style ($, 4dp,
  runtime). Raw None becomes an em dash; a raw float/int gets a
  sensible default (4dp / thousands-separated) so a forgotten formatter still
  produces a readable cell rather than a bare 0.8765432109.
- No nested dicts/lists are rendered — pass scalars or pre-rendered strings.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

# Em dash — the single canonical "missing value" marker for every table.
EMDASH = "—"

# Metric precision (accuracy / mcc / f1 / precision / recall / auroc / loss).
_METRIC_PLACES = 4


def fmt_metric(value: Any, places: int = _METRIC_PLACES) -> str:
    """Format a metric to places decimals; None/non-numeric → em dash."""
    if value is None:
        return EMDASH
    try:
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return EMDASH


def fmt_loss(value: Any) -> str:
    """Loss to 4 decimals; None/non-numeric → em dash."""
    return fmt_metric(value, _METRIC_PLACES)


def fmt_cost(value: Any) -> str:
    """USD cost as $X.XX (2 decimals); None/non-numeric → em dash."""
    if value is None:
        return EMDASH
    try:
        return f"${float(value):.2f}"
    except (TypeError, ValueError):
        return EMDASH


def fmt_cost_precise(value: Any, places: int = 4) -> str:
    """USD cost at higher precision (default 4dp) for sub-cent agent costs."""
    if value is None:
        return EMDASH
    try:
        return f"${float(value):.{places}f}"
    except (TypeError, ValueError):
        return EMDASH


def fmt_int(value: Any) -> str:
    """Integer with thousands separators; None/non-numeric → em dash."""
    if value is None:
        return EMDASH
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return EMDASH


def fmt_runtime(seconds: Any) -> str:
    """Human-readable duration from a seconds value.

    Examples: 45.0 → "45s", 95 → "1m 35s", 3725 → "1h 2m 5s", 0 → "0s".
    None/negative/non-numeric → em dash.
    """
    if seconds is None:
        return EMDASH
    try:
        total = float(seconds)
    except (TypeError, ValueError):
        return EMDASH
    if total < 0:
        return EMDASH

    total_int = int(round(total))
    hours, rem = divmod(total_int, 3600)
    minutes, secs = divmod(rem, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    return " ".join(parts)


def fmt_value(value: Any, places: int = _METRIC_PLACES) -> str:
    """Default cell formatter: None → em dash, float → places dp,
    bool/int → thousands-separated int, everything else → str."""
    if value is None:
        return EMDASH
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{places}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _cell(value: Any) -> str:
    """Render one cell, escaping markdown-table-breaking characters."""
    text = fmt_value(value)
    # A literal pipe would split the cell; newlines would break the row.
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").strip()


_ALIGN_SEP = {
    "left": ":---",
    "right": "---:",
    "center": ":--:",
}


def markdown_table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    *,
    headers: Mapping[str, str] | None = None,
    align: Mapping[str, str] | None = None,
) -> str:
    """Render rows as a GitHub-flavored markdown table.

    Args:
        rows: sequence of mappings; missing keys / None render as em dash.
        columns: column keys in stable display order.
        headers: optional column-key → header-label overrides (default: the key).
        align: optional column-key → "left"|"right"|"center" (default left).

    Returns the table as a string (no trailing newline). With no columns the
    result is an empty string; with columns but no rows, just the header +
    separator (so the section still reads as "a table, currently empty").
    """
    if not columns:
        return ""
    headers = headers or {}
    align = align or {}

    header_cells = [str(headers.get(col, col)) for col in columns]
    sep_cells = [_ALIGN_SEP.get(align.get(col, "left"), _ALIGN_SEP["left"]) for col in columns]

    lines = [
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join(sep_cells) + " |",
    ]
    for row in rows:
        cells = [_cell(row.get(col)) for col in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

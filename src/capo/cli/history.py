"""Run history table — capo history. Scans runs_root/*/state.json, newest first.

A long dataset ref used to widen its column until it crowded out every other
column; each column is now hard-capped and truncated with an ellipsis so the
table always fits the terminal. The full, untruncated values are available via
``capo history --full <run_id>`` / ``capo inspect <run_id>`` (print_run_detail).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from rich import box
from rich.table import Table
from rich.text import Text

from .colors import console

# per-column character caps (the run id is left uncapped: it is the actionable
# handle for capo health/resume/inspect, so it must stay copyable in full).
# dataset/model are kept tight so the narrow-but-critical cost column is never
# squeezed off the right edge — full values live in `capo inspect <run_id>`.
_W_DATASET = 24
_W_MODEL = 18
_W_STRATEGY = 14
_W_PHASE = 18
# cost is pinned to a fixed width (allocated before any flexible column), so it
# always renders in full no matter how long the dataset/model fields get.
_W_COST = 8


def ellipsize(value: object, max_width: int) -> str:
    """Truncate value to max_width characters, marking the cut with an ellipsis."""
    value = str(value or "")
    if max_width <= 0 or len(value) <= max_width:
        return value
    return value[: max_width - 1] + "…"


def _cost_str(run_dir: Path) -> str:
    """Actual cost (post-run) if available, else projected (pre-launch)."""
    final = run_dir / "reports" / "final_summary.json"
    if final.exists():
        try:
            v = json.loads(final.read_text(encoding="utf-8")).get("actual_cost_usd")
            if v is not None:
                return f"{v:.2f}"
        except (OSError, json.JSONDecodeError):
            pass
    cost = run_dir / "pricing" / "cost_report.json"
    if cost.exists():
        try:
            v = (json.loads(cost.read_text(encoding="utf-8")).get("projections") or {}).get(
                "projected_cost_usd"
            )
            if v is not None:
                return f"~{v:.2f}"
        except (OSError, json.JSONDecodeError):
            pass
    return ""


def _phase_text(phase: str, terminal: str | None) -> Text:
    if phase == "completed" and terminal == "completed":
        txt = Text("completed ✓", style="table.done")
    elif terminal == "failed" or phase == "failed":
        txt = Text("failed ✗", style="table.fail")
    elif phase in ("pre_launch", "training", "finalizing"):
        txt = Text(f"{phase} ●", style="table.run")
    else:
        txt = Text(phase, style="muted")
    if terminal and terminal != phase and not (phase == "completed" and terminal == "completed"):
        txt.append(f" ({terminal})", style="muted")
    return txt


def print_history(runs_root: Path, limit: int = 20) -> None:
    state_files = sorted(
        runs_root.glob("*/state.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]

    if not state_files:
        console.print(f"[muted]No runs found under {runs_root}.[/]")
        return

    t = Table(
        box=box.SIMPLE_HEAD,
        header_style="table.header",
        border_style="brand.dim",
        show_lines=False,
        padding=(0, 1),
    )
    # no_wrap + overflow="ellipsis" + max_width keep every column bounded, so one
    # long field (a dataset ref) can never widen its column and crowd out the rest.
    t.add_column("Run ID", style="table.id", no_wrap=True)
    t.add_column("Dataset", style="brand.dim", no_wrap=True, overflow="ellipsis", max_width=_W_DATASET)
    t.add_column("Model", style="muted", no_wrap=True, overflow="ellipsis", max_width=_W_MODEL)
    t.add_column("Strategy", style="muted", no_wrap=True, overflow="ellipsis", max_width=_W_STRATEGY)
    t.add_column("Phase", no_wrap=True, overflow="ellipsis", max_width=_W_PHASE)
    # fixed width (not max_width): a fixed column is reserved before flexible ones
    # are sized, so cost is guaranteed to show even on a narrow terminal.
    t.add_column("Cost $", style="metric.key", no_wrap=True, justify="right", width=_W_COST)
    t.add_column("Started", style="muted", no_wrap=True)

    for sf in state_files:
        try:
            d = json.loads(sf.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        created = d.get("created_at", "")
        try:
            ts = datetime.fromisoformat(created.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            ts = created[:16]
        model = (d.get("model_id") or "").split("/")[-1]
        t.add_row(
            d.get("run_id", ""),
            d.get("dataset_ref", ""),
            model,
            d.get("fine_tune_strategy", ""),
            _phase_text(d.get("current_phase", "?"), d.get("terminal_state")),
            _cost_str(sf.parent),
            ts,
        )

    console.print()
    console.print(t)
    console.print(
        f"  [muted]Showing {len(state_files)} most recent runs  ·  runs root: {runs_root}[/]\n"
        "  [muted]Full detail for one run: [/][cmd]capo inspect <run_id>[/]"
        "[muted]  ·  [/][cmd]capo history --full <run_id>[/]\n"
    )


# === single-run detail (capo history --full / capo inspect) ===
# the table truncates; this prints the full, untruncated values for one run, so
# a long dataset ref or model path is never hidden. inspect adds an artifact list.

_INSPECT_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("task.md", "task brief"),
    ("profile/profile.json", "dataset profile"),
    ("reports/model_selection.json", "model + strategy choice"),
    ("infra.json", "GPU / instance"),
    ("pricing/cost_report.json", "cost gate"),
    ("probe/probe_result.json", "feasibility probe"),
    ("reports/handoff.json", "training handoff"),
    ("reports/final_summary.json", "final summary"),
    ("RUN_REPORT.md", "scientific report"),
    ("outputs/metrics.jsonl", "training metrics"),
    ("outputs/run.log", "run log"),
)


def print_run_detail(runs_root: Path, run_id: str, *, list_artifacts: bool = False) -> None:
    """Full, untruncated key/value detail for one run (capo history --full / inspect).

    list_artifacts=True additionally lists which run-dir artifacts are present and
    their sizes — the inspect view. Reads the same state.json the table reads; no
    run behaviour is touched."""
    from rich.panel import Panel

    run_dir = runs_root / run_id
    state_path = run_dir / "state.json"
    if not state_path.exists():
        console.print(f"\n  [err]Not found:[/] {run_id}  [muted](no state.json under {run_dir})[/]\n")
        return
    try:
        d = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        console.print(f"\n  [err]Unreadable state.json for {run_id}:[/] {exc}\n")
        return

    created = d.get("created_at", "")
    try:
        ts = datetime.fromisoformat(created.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        ts = created[:19]

    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column("Key", style="metric.key", no_wrap=True)
    t.add_column("Value", style="brand.dim", overflow="fold")  # fold = wrap, never truncate
    t.add_row("Run ID", run_id)
    t.add_row("Dataset", d.get("dataset_ref", "") or "—")
    t.add_row("Model", d.get("model_id", "") or "—")
    t.add_row("Strategy", d.get("fine_tune_strategy", "") or "—")
    t.add_row("Phase", _phase_text(d.get("current_phase", "?"), d.get("terminal_state")))
    t.add_row("Cost $", _cost_str(run_dir) or "—")
    t.add_row("Started", ts or "—")
    if d.get("error"):
        t.add_row("Error", Text(str(d["error"]), style="err"))
    t.add_row("Run dir", str(run_dir))
    t.add_row("Health", Text.from_markup(f"[cmd]capo health {run_id}[/]"))

    console.print()
    console.print(Panel(t, title=f"[brand] Run {run_id}[/]", border_style="brand.dim", padding=(0, 1)))

    if list_artifacts:
        a = Table(show_header=False, box=None, padding=(0, 2))
        a.add_column("", no_wrap=True, width=3)
        a.add_column("Artifact", style="brand.dim", no_wrap=True)
        a.add_column("", style="muted")
        for rel, label in _INSPECT_ARTIFACTS:
            p = run_dir / rel
            if p.exists():
                size = p.stat().st_size
                a.add_row("[ok]✓[/]", rel, f"{label}  ·  {size / 1024:.1f} KB" if size else label)
            else:
                a.add_row("[muted]·[/]", f"[muted]{rel}[/]", f"[muted]{label} — missing[/]")
        console.print()
        console.print("  [metric.key]Artifacts[/]")
        console.print(a)
    console.print()

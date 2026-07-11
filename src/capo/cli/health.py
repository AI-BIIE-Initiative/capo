"""
Health card for a single run — capo health <id> and the /health console command.

Reads the real on-disk artifacts a run produces:
  state.json                     current_phase / terminal_state / pause state
  outputs/status.json            training-side state / step / epoch (synced back)
  reports/health/history.jsonl   Haiku monitor reports (metrics, trend, gpu, summary)
  reports/final_summary.json     terminal metrics + actual cost (post-run)
  pricing/cost_report.json       total_steps + projected cost (pre-launch)
Every row degrades gracefully: a missing file just drops its rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .colors import console


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_history_tail(run_dir: Path) -> tuple[dict, dict]:
    """Return (latest, previous) entries from reports/health/history.jsonl."""
    p = run_dir / "reports" / "health" / "history.jsonl"
    entries: list[dict] = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    latest = entries[-1] if entries else {}
    prev = entries[-2] if len(entries) >= 2 else {}
    return latest, prev


def _trend_style(cur: Optional[float], prev: Optional[float], higher_better: bool) -> str:
    if cur is None:
        return "muted"
    if prev is None:
        return "metric.key"
    improving = cur >= prev if higher_better else cur <= prev
    return "metric.good" if improving else "metric.warn"


def _gpu_style(pct: Optional[float]) -> str:
    if pct is None:
        return "muted"
    if pct > 95:
        return "metric.bad"
    if pct > 80:
        return "metric.warn"
    return "metric.good"


_SEVERITY_STYLE = {"info": "metric.good", "warning": "metric.warn", "severe": "metric.bad"}


def print_health_card(run_id: str, runs_root: Path) -> None:
    run_dir = runs_root / run_id
    if not run_dir.exists():
        console.print(f"[err]Run not found:[/] {run_id}  [muted]({runs_root})[/]")
        return

    state = _read_json(run_dir / "state.json")
    status = _read_json(run_dir / "outputs" / "status.json")
    final = _read_json(run_dir / "reports" / "final_summary.json")
    cost = _read_json(run_dir / "pricing" / "cost_report.json")
    latest, prev = _read_history_tail(run_dir)
    metrics = latest.get("metrics") or {}
    prev_metrics = prev.get("metrics") or {}

    phase = state.get("current_phase", "unknown")
    terminal = state.get("terminal_state")
    run_state = latest.get("state") or status.get("state") or phase

    t = Table(show_header=False, box=None, padding=(0, 3))
    t.add_column("k", style="metric.key", no_wrap=True, min_width=14)
    t.add_column("v")

    def row(k: str, v: str, style: str = "") -> None:
        t.add_row(k, f"[{style}]{v}[/]" if style else v)

    # --- status ---
    if phase == "completed" and terminal == "completed":
        row("Status", "completed ✓", "phase.done")
    elif terminal == "failed" or phase == "failed":
        row("Status", f"{phase} ✗" + (f"  ({terminal})" if terminal else ""), "phase.fail")
    elif phase in ("pre_launch", "training", "finalizing"):
        row("Status", f"{phase} ●", "phase.run")
    else:
        row("Status", str(run_state), "muted")

    if state.get("paused"):
        row("Paused", state.get("pause_reason") or "awaiting user", "metric.warn")

    # --- progress (step / total_steps from cost_report) ---
    step = latest.get("step") or status.get("step")
    epoch = latest.get("epoch") or status.get("epoch")
    total_steps = (cost.get("training_plan") or {}).get("total_steps")
    if step is not None and total_steps:
        pct = min(100.0, step / total_steps * 100)
        filled = int(20 * pct / 100)
        bar = "█" * filled + "░" * (20 - filled)
        row("Progress", f"[phase.done]{bar}[/] {step:,} / {total_steps:,}  ({pct:.0f}%)")
    elif step is not None:
        extra = f"  epoch {epoch}" if epoch is not None else ""
        row("Progress", f"step {step:,}{extra}")
    else:
        row("Progress", "not started", "muted")

    # --- metrics (final_summary wins post-run; else latest monitor report) ---
    fm = final.get("final_metrics") or {}

    def metric_row(label: str, key: str, higher_better: bool, fmt: str = "{:.4f}") -> None:
        cur = fm.get(key, metrics.get(key))
        if cur is None:
            return
        style = _trend_style(metrics.get(key), prev_metrics.get(key), higher_better)
        try:
            row(label, fmt.format(cur), style)
        except (ValueError, TypeError):
            row(label, str(cur), style)

    metric_row("Train loss", "train_loss", higher_better=False)
    metric_row("Val loss", "val_loss", higher_better=False)
    metric_row("Val MCC", "val_mcc", higher_better=True)
    metric_row("Val AUROC", "val_auroc", higher_better=True)
    # test metrics only appear post-run in final_summary
    if "test_mcc" in fm:
        row("Test MCC", f"{fm['test_mcc']:.4f}", "metric.good")
    if "test_auroc" in fm:
        row("Test AUROC", f"{fm['test_auroc']:.4f}", "metric.good")

    # --- trend / severity / gpu ---
    if latest.get("trend"):
        row("Trend", str(latest["trend"]),
            "metric.good" if latest["trend"] == "improving" else "metric.warn")
    sev = latest.get("severity")
    if sev:
        row("Severity", sev, _SEVERITY_STYLE.get(sev, "muted"))
    alerts = latest.get("alerts") or []
    if alerts:
        row("Alerts", "; ".join(str(a) for a in alerts)[:80], "metric.warn")
    gpu_util = latest.get("gpu_util_pct")
    gpu_mem = latest.get("gpu_mem_pct")
    if gpu_util is not None or gpu_mem is not None:
        row("GPU", f"util {gpu_util or 0}%  ·  mem {gpu_mem or 0}%", _gpu_style(gpu_util))

    # --- cost ---
    actual = final.get("actual_cost_usd")
    projected = (cost.get("projections") or {}).get("projected_cost_usd")
    itype = cost.get("instance_type")
    if actual is not None:
        row("Cost", f"${actual:.2f} actual" + (f"  ({itype})" if itype else ""), "metric.key")
    elif projected is not None:
        row("Cost", f"${projected:.2f} projected" + (f"  ({itype})" if itype else ""), "metric.key")

    # --- trackio ---
    trackio = final.get("trackio_url") or latest.get("trackio_url")
    if trackio:
        row("Trackio", trackio, "metric.key")

    # --- latest monitor summary (natural language) ---
    summary = latest.get("summary")
    if summary:
        t.add_row("Note", Text(summary, style="brand.dim"))

    title = Text.assemble((" Training Health", "brand"), ("  ·  ", "brand.dim"), (run_id, "muted"))
    console.print(Panel(t, title=title, border_style="brand.dim", padding=(0, 2)))

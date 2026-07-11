"""Tests for the health monitor's deterministic billing-critical backstops.

The monitor must catch a silent crash that leaves the GPU idle behind a stale
status.json WITHOUT false-positiving a healthy
GPU-bound phase that simply hasn't written metrics yet (e.g. boltz embedding).

We test ``_deadline_escalation`` directly via ``object.__new__`` so the test
never constructs an AgentRunner / touches the API.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from capo.observability.training_health_monitor import (
    HealthReport,
    TrainingHealthMonitor,
    _parse_iso,
)


def _monitor(deadline_iso=None, launched_iso=None):
    mon = object.__new__(TrainingHealthMonitor)
    mon._gpu_active_deadline_iso = deadline_iso
    mon._handoff = {"launched_at_iso": launched_iso} if launched_iso else {}
    # State the decision methods read (mirror __init__).
    mon._last_report = None
    mon._heartbeat_timeout_sec = TrainingHealthMonitor.HEARTBEAT_TIMEOUT_DEFAULT_SEC
    mon._stall_streak = 0
    mon._soft_severe_streak = 0
    mon._deadline_grace_used = 0.0
    mon._deadline_noprogress_streak = 0
    return mon


def _report(**kw):
    base = dict(
        ts="t", state="running", gpu_util_pct=0, metrics_rows=0,
        metrics={}, step=None, status_age_sec=4000,
    )
    base.update(kw)
    return HealthReport(**base)


def test_escalates_idle_gpu_past_deadline_no_progress():
    now = datetime.now(timezone.utc)
    mon = _monitor(deadline_iso=(now - timedelta(minutes=30)).isoformat())
    handoff = mon._deadline_escalation(_report())
    assert handoff is not None
    assert handoff.kind == "escalation"
    assert "deadline" in handoff.reason


def test_does_not_escalate_when_gpu_busy():
    # Healthy boltz embed phase: GPU pegged, no metrics rows yet.
    now = datetime.now(timezone.utc)
    mon = _monitor(deadline_iso=(now - timedelta(minutes=30)).isoformat())
    assert mon._deadline_escalation(_report(gpu_util_pct=98)) is None


def test_does_not_escalate_when_metrics_flowing():
    now = datetime.now(timezone.utc)
    mon = _monitor(deadline_iso=(now - timedelta(minutes=30)).isoformat())
    rep = _report(metrics_rows=12, step=300, metrics={"val_loss": 0.4})
    assert mon._deadline_escalation(rep) is None


def test_does_not_escalate_before_deadline():
    now = datetime.now(timezone.utc)
    mon = _monitor(deadline_iso=(now + timedelta(minutes=10)).isoformat())
    assert mon._deadline_escalation(_report()) is None


def test_terminal_states_flow_through():
    now = datetime.now(timezone.utc)
    mon = _monitor(deadline_iso=(now - timedelta(minutes=30)).isoformat())
    assert mon._deadline_escalation(_report(state="completed")) is None
    assert mon._deadline_escalation(_report(state="failed")) is None


def test_no_gpu_reading_is_not_escalated():
    now = datetime.now(timezone.utc)
    mon = _monitor(deadline_iso=(now - timedelta(minutes=30)).isoformat())
    assert mon._deadline_escalation(_report(gpu_util_pct=None)) is None


def test_fallback_deadline_from_launched_at():
    now = datetime.now(timezone.utc)
    mon = _monitor(launched_iso=(now - timedelta(minutes=200)).isoformat())
    # No explicit deadline -> launched_at + DEFAULT_GPU_ACTIVE_BUDGET_SEC (90m),
    # which 200 min ago is well past -> escalates.
    assert mon._gpu_active_deadline() is not None
    assert mon._deadline_escalation(_report()) is not None


def test_no_deadline_no_launch_is_safe():
    mon = _monitor()
    assert mon._gpu_active_deadline() is None
    assert mon._deadline_escalation(_report()) is None


def test_parse_iso_variants():
    assert _parse_iso("2026-06-24T23:04:39Z") is not None
    assert _parse_iso("2026-06-24T23:04:39+00:00") is not None
    assert _parse_iso(None) is None
    assert _parse_iso("not-a-date") is None


# ---------------------------------------------------------------------------
# Forward-progress false-positive protection (the ClinVar CPU-baseline regression)
# ---------------------------------------------------------------------------

def _step(mon, report):
    """Mimic run_loop: classify vs the PREVIOUS report, then advance _last_report."""
    verdict = mon._classify_handoff(report)
    mon._last_report = report
    return verdict


def test_does_not_escalate_live_progressing_run_past_deadline():
    # A slow CPU baseline past its (too-tight) deadline: GPU idle, no metrics,
    # but status FRESH and stdout ADVANCING each tick. Must NOT be escalated —
    # the monitor grants grace instead of killing a live, working run.
    now = datetime.now(timezone.utc)
    mon = _monitor(deadline_iso=(now - timedelta(minutes=30)).isoformat())
    escalated = False
    for line in ("C=0.01", "C=0.1", "C=1.0", "C=10.0", "done"):
        rep = _report(status_age_sec=120, last_stdout_line=line,
                      alerts=["stalled", "gpu_cold_no_progress"], severity="severe")
        if _step(mon, rep) is not None:
            escalated = True
            break
    assert not escalated


def test_escalates_frozen_run_after_debounce_not_first_tick():
    # A genuinely hung run (fresh status but stdout FROZEN) is escalated — but
    # only after DEADLINE_ESCALATE_TICKS, never on the first observation.
    now = datetime.now(timezone.utc)
    mon = _monitor(deadline_iso=(now - timedelta(minutes=30)).isoformat())
    outcomes = []
    for _ in range(TrainingHealthMonitor.DEADLINE_ESCALATE_TICKS + 2):
        rep = _report(status_age_sec=60, last_stdout_line="FROZEN",
                      alerts=[], severity="info")
        outcomes.append(_step(mon, rep) is not None)
        if outcomes[-1]:
            break
    assert outcomes[0] is False          # never on the first tick
    assert True in outcomes              # but eventually escalates


def test_stale_status_escalates_without_grace():
    # A stale status.json is the classic dead-behind-frozen-status signal and
    # must escalate even though it is the first tick (grace requires fresh status).
    now = datetime.now(timezone.utc)
    mon = _monitor(deadline_iso=(now - timedelta(minutes=30)).isoformat())
    rep = _report(status_age_sec=5000, last_stdout_line="x")
    assert _step(mon, rep) is not None


def test_soft_severe_debounced_but_hard_severe_immediate():
    # No deadline set -> the deadline backstop is inert; exercise the severity path.
    # A HARD severe alert (cuda_oom) escalates on the first tick.
    mon = _monitor()
    hard = _report(gpu_util_pct=80, metrics_rows=50, last_stdout_line="oom",
                   alerts=["cuda_oom"], severity="severe")
    assert _step(mon, hard) is not None

    # A SOFT severe alert (gpu_cold/stalled) with forward progress does NOT escalate.
    mon = _monitor()
    for line in ("a", "b", "c", "d"):
        soft = _report(gpu_util_pct=0, last_stdout_line=line,
                       alerts=["gpu_cold_no_progress"], severity="severe")
        assert _step(mon, soft) is None

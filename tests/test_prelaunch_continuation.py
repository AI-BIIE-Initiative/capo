"""Regression tests for the Phase A pre-launch continuation loop.

The failure this locks down: run binding-boltz-20260703-0034-aa91 ran the whole
pre-launch phase in ONE agent query; the agent ended its turn (a clean
ResultMessage, subtype="success") WHILE WAITING on a boltz warm-up, before it
ever launched training. The orchestrator had no continuation — it hit the
`else: state = "unknown"` dead end and quit. Training never launched, no
handoff.json was written, and the run was silently lost.

The fix: after each Phase A agent turn, classify the outcome. If the agent
neither launched training (handoff.json) nor aborted (an abort marker), it is
"stuck" — re-prompt it to resume, bounded by max_prelaunch_continuations. If it
is still stuck after the budget is spent, the run is `failed` (with a
diagnosable stall marker), never a silent `unknown`.
"""

from __future__ import annotations

import json

from capo.orchestration.fine_tuning_orchestrator import FineTuningOrchestrator as FTO


# ---------------------------------------------------------------------------
# _prelaunch_outcome — the loop's stop/continue decision
# ---------------------------------------------------------------------------

def test_prelaunch_outcome_launched_when_handoff_exists(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    handoff = reports / "handoff.json"
    handoff.write_text("{}", encoding="utf-8")
    assert FTO._prelaunch_outcome(reports, handoff, is_resume=False) == "launched"


def test_prelaunch_outcome_aborted_when_marker_exists(tmp_path):
    # A cost-gate abort is a LEGITIMATE terminal stop — the loop must not keep
    # re-prompting a run the agent correctly aborted.
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "abort_over_budget.json").write_text("{}", encoding="utf-8")
    handoff = reports / "handoff.json"  # does not exist
    assert FTO._prelaunch_outcome(reports, handoff, is_resume=False) == "aborted"


def test_prelaunch_outcome_stuck_when_neither(tmp_path):
    # The binding-boltz case: agent ended its turn mid-flight — no handoff, no
    # abort marker. This MUST be "stuck" (→ continue), never a silent terminal.
    reports = tmp_path / "reports"
    reports.mkdir()
    handoff = reports / "handoff.json"  # does not exist
    assert FTO._prelaunch_outcome(reports, handoff, is_resume=False) == "stuck"


# ---------------------------------------------------------------------------
# _write_prelaunch_stall_marker — the diagnosable artifact behind a `failed`
# ---------------------------------------------------------------------------

def test_stall_marker_records_reason_and_subtype(tmp_path):
    reports = tmp_path / "reports"
    reports.mkdir()
    FTO._write_prelaunch_stall_marker(reports, subtype="success", continuations=3)
    marker = reports / "prelaunch_stall.json"
    assert marker.exists()
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["state"] == "prelaunch_stalled"
    assert payload["last_agent_subtype"] == "success"
    assert payload["continuations_used"] == 3
    assert "handoff.json" in payload["reason"]


def test_stall_marker_creates_reports_dir_if_missing(tmp_path):
    # Best-effort: even if reports/ does not exist yet, the marker is written.
    reports = tmp_path / "reports"  # not created
    FTO._write_prelaunch_stall_marker(reports, subtype=None, continuations=0)
    assert (reports / "prelaunch_stall.json").exists()


# ---------------------------------------------------------------------------
# The continuation prompt — must drive the agent to a terminal artifact
# ---------------------------------------------------------------------------

def test_continuation_prompt_is_a_resume_not_a_restart():
    p = FTO._PRELAUNCH_CONTINUATION_PROMPT.format(
        run_id="binding-boltz-x",
        local_run_dir="/local/capo/binding-boltz-x",
        remote_run_dir="~/capo_runs/binding-boltz-x",
    )
    # names the run and both dirs
    assert "binding-boltz-x" in p
    assert "/local/capo/binding-boltz-x" in p
    assert "~/capo_runs/binding-boltz-x" in p
    # forces the two terminal artifacts
    assert "handoff.json" in p
    assert "abort marker" in p
    # tells it to RESUME (reuse on-disk + remote state), not start over
    assert "RESUME" in p
    assert "do not restart from scratch" in p.lower()
    # explicitly forbids the fragile blocking waiter that stalled the first run
    assert "blocking foreground waiter" in p.lower()
    assert "do not end your turn" in p.lower()


# ---------------------------------------------------------------------------
# _read_log_tail — the recovery agent's evidence must be the REMOTE TRAINING
# log, never the orchestrator's own run.log (that would mislead diagnosis).
# ---------------------------------------------------------------------------

def test_read_log_tail_prefers_remote_training_log(tmp_path):
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "train.log").write_text("step 850 loss 0.12\nCUDA out of memory\n", encoding="utf-8")
    tail = FTO._read_log_tail(tmp_path)
    assert "CUDA out of memory" in tail


def test_read_log_tail_never_returns_orchestrator_run_log(tmp_path):
    # Only run.log exists (the agent's OWN chatter). It must NOT be surfaced as
    # a training log — the recovery agent would diagnose the wrong thing.
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "run.log").write_text("01:13 [status] waiting on warm-up co-fold\n", encoding="utf-8")
    tail = FTO._read_log_tail(tmp_path)
    assert "warm-up" not in tail
    assert "no local training log" in tail


def test_read_log_tail_skips_empty_train_log_for_stderr(tmp_path):
    # An empty train.log must not shadow a real traceback in train_err.log.
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "train.log").write_text("", encoding="utf-8")
    (outputs / "train_err.log").write_text("Traceback ...\nRuntimeError: boom\n", encoding="utf-8")
    tail = FTO._read_log_tail(tmp_path)
    assert "RuntimeError: boom" in tail

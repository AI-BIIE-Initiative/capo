"""Tests for capo.orchestration.training_recovery — safe-fix gate, ledger, report."""

from __future__ import annotations

import json

from capo.orchestration.training_recovery import (
    RecoveryAttempt,
    RecoveryLedger,
    build_recovery_context,
    classify_fix_safety,
    render_recovery_report_markdown,
)


# ---------------------------------------------------------------------------
# safe / unsafe classification
# ---------------------------------------------------------------------------

def test_safe_fix_types_classified_safe():
    for ft in ("reduce_batch_size", "enable_grad_checkpointing", "install_dependency"):
        assert classify_fix_safety(ft) == "safe"


def test_unsafe_fix_types_classified_unsafe():
    for ft in ("change_model", "change_dataset", "delete_data"):
        assert classify_fix_safety(ft) == "unsafe"


def test_unknown_fix_is_unsafe_by_default():
    assert classify_fix_safety("frobnicate_the_widget") == "unsafe"
    assert classify_fix_safety(None) == "unsafe"


def test_disable_plotting_is_safe_and_auto_applied():
    # A cosmetic plotting crash must be auto-recoverable: disabling inline plots
    # changes nothing scientific (finalizer regenerates PNGs from the CSVs), so
    # the loop applies it without stopping for user confirmation.
    assert classify_fix_safety("disable_plotting") == "safe"
    ra = RecoveryAttempt.from_verdict(
        1,
        {
            "failure_category": "plotting_bug",
            "fix_type": "disable_plotting",
            "fix_applied": "added --no-inline-plots to launch_command.sh",
            "relaunched": True,
            "new_pid": 4242,
            "outcome": "applied",
        },
    )
    assert ra.fix_safety == "safe"
    assert ra.outcome == "applied"       # NOT needs_user — no user prompt over a plot bug


# ---------------------------------------------------------------------------
# RecoveryAttempt.from_verdict — classifier is authoritative
# ---------------------------------------------------------------------------

def test_from_verdict_safe_oom_fix():
    a = RecoveryAttempt.from_verdict(
        1,
        {
            "failure_category": "oom",
            "diagnosis": "CUDA OOM at batch 4",
            "fix_type": "reduce_batch_size",
            "fix_applied": "batch_size 32 → 16",
            "canary_rerun": True,
            "relaunched": True,
            "new_pid": 4242,
            "outcome": "recovered",
        },
    )
    assert a.fix_safety == "safe"
    assert a.relaunched is True
    assert a.new_pid == 4242
    assert a.outcome == "recovered"


def test_from_verdict_agent_cannot_relabel_unsafe_as_safe():
    # Even if the agent claims safety, the classifier (authoritative) overrides.
    a = RecoveryAttempt.from_verdict(
        1, {"fix_type": "change_model", "fix_safety": "safe", "outcome": "applied"}
    )
    assert a.fix_safety == "unsafe"


def test_from_verdict_none():
    a = RecoveryAttempt.from_verdict(2, None)
    assert a.attempt == 2 and a.outcome == "failed"


# ---------------------------------------------------------------------------
# RecoveryLedger
# ---------------------------------------------------------------------------

def test_ledger_to_dict_and_recovered():
    ledger = RecoveryLedger(
        attempts=[
            RecoveryAttempt(attempt=1, outcome="failed"),
            RecoveryAttempt(attempt=2, outcome="recovered", fix_type="reduce_batch_size"),
        ],
        final_outcome="recovered",
    )
    d = ledger.to_dict()
    assert d["n_attempts"] == 2
    assert d["final_outcome"] == "recovered"
    assert ledger.recovered is True


def test_ledger_write_json(tmp_path):
    ledger = RecoveryLedger(attempts=[RecoveryAttempt(attempt=1)], final_outcome="exhausted")
    p = tmp_path / "recovery_ledger.json"
    ledger.write_json(p)
    loaded = json.loads(p.read_text(encoding="utf-8"))
    assert loaded["final_outcome"] == "exhausted"


# ---------------------------------------------------------------------------
# report rendering
# ---------------------------------------------------------------------------

def test_render_recovery_report():
    ledger = RecoveryLedger(
        attempts=[
            RecoveryAttempt(
                attempt=1, failure_category="oom", diagnosis="OOM",
                fix_type="reduce_batch_size", fix_safety="safe",
                fix_applied="bs 32→16", canary_rerun=True, outcome="recovered",
            )
        ],
        final_outcome="recovered",
    )
    md = render_recovery_report_markdown(ledger)
    assert "## Training Recovery" in md
    assert "| Attempt | Failure | Diagnosis | Fix Applied | Safety | Canary Rerun | Outcome |" in md
    assert "reduce_batch_size" in md or "bs 32" in md
    assert "recovered" in md


def test_render_recovery_report_needs_user():
    ledger = RecoveryLedger(
        attempts=[RecoveryAttempt(attempt=1, fix_type="change_model", outcome="needs_user")],
        final_outcome="needs_user",
    )
    md = render_recovery_report_markdown(ledger)
    assert "UNSAFE" in md and "user confirmation" in md


# ---------------------------------------------------------------------------
# context builder
# ---------------------------------------------------------------------------

def test_build_recovery_context_includes_key_fields():
    ctx = build_recovery_context(
        attempt=2,
        max_attempts=3,
        failure={"failure_category": "oom", "summary": "OOM at step 5"},
        config={"batch_size": 32},
        last_checkpoint="checkpoints/last",
        canary_passed=True,
        log_tail="CUDA out of memory",
        ssh_alias="ubuntu@1.2.3.4",
        remote_run_dir="~/capo_runs/r1",
        local_run_dir="/local/r1",
    )
    assert "recovery_attempt = 2 of 3" in ctx
    assert "canary_passed = True" in ctx
    assert "failure_category = oom" in ctx
    assert "reduce_batch_size" in ctx  # safe-fix menu present
    assert "change_model" in ctx       # unsafe-fix menu present
    assert "CUDA out of memory" in ctx

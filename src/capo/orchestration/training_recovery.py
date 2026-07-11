"""Intelligent post-canary training recovery — state, safe-fix gate, reporting.

When training fails *after* the canary/probe has passed, CAPO does not simply
abort. The orchestrator runs a bounded agentic recovery loop:

    failure detected → monitor halts → Sonnet diagnosis agent receives the
    failure context → proposes a fix → SAFE fixes are applied automatically
    (rerunning the canary when needed) and training relaunched → re-monitor →
    repeat up to max_training_recovery_attempts → final report.

UNSAFE fixes (those that change scientific meaning or are hard to reverse) are
never auto-applied — the loop stops and asks the user.

This module holds the pure, testable pieces: the per-attempt record, the
ledger, the safe/unsafe classifier, the diagnosis-prompt builder, and the
"## Training Recovery" report renderer. The orchestrator owns the agent
dispatch + remote relaunch + re-monitor.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from capo.report.tables import EMDASH, markdown_table

# Fix categories the loop may auto-apply — reversible config / hyperparameter /
# dependency changes that do not alter the scientific question.
SAFE_FIX_TYPES: frozenset[str] = frozenset(
    {
        "reduce_batch_size",
        "increase_grad_accum",
        "enable_grad_checkpointing",
        "reduce_seq_length",
        "lower_learning_rate",
        "set_precision",            # e.g. bf16/fp16/fp32 toggle
        "install_dependency",
        "disable_custom_kernel",    # e.g. --no_kernels fallback
        "fix_column_mapping",       # map sequence/label column name
        "set_env_var",
        "reduce_num_workers",
        "pin_dataset_revision",
        "reduce_eval_batch_size",
        "clip_grad_norm",
        "disable_plotting",         # skip inline plots; finalizer regenerates from CSVs
    }
)

# Fix categories that change the experiment's meaning or are hard to reverse —
# never auto-applied; the loop stops and asks the user.
UNSAFE_FIX_TYPES: frozenset[str] = frozenset(
    {
        "change_model",
        "change_dataset",
        "change_task",
        "change_labels",
        "modify_training_logic",
        "increase_budget",
        "delete_data",
        "terminate_instance",
        "change_split",
    }
)


def classify_fix_safety(fix_type: str | None) -> str:
    """Classify a proposed fix as "safe" or "unsafe".

    Unknown / missing fix types are treated as **unsafe** (conservative: a fix
    we can't recognize must be confirmed by the user, never auto-applied)."""
    if fix_type and fix_type in SAFE_FIX_TYPES:
        return "safe"
    return "unsafe"


@dataclass
class RecoveryAttempt:
    """One diagnose → fix → relaunch → re-monitor cycle."""

    attempt: int
    failure_category: str = "unknown"
    diagnosis: str = ""
    fix_type: str = ""
    fix_safety: str = "unsafe"          # safe | unsafe
    fix_applied: str = ""               # human description of what changed
    canary_rerun: bool = False
    relaunched: bool = False
    new_pid: int | None = None
    outcome: str = "failed"             # recovered | failed | needs_user | applied | resume_monitoring
    evidence: str = ""

    @classmethod
    def from_verdict(cls, attempt: int, verdict: dict | None) -> "RecoveryAttempt":
        """Build from the recovery agent's structured JSON verdict."""
        if not verdict:
            return cls(attempt=attempt, outcome="failed", diagnosis="no verdict produced")
        fix_type = str(verdict.get("fix_type") or "")
        outcome = str(verdict.get("outcome") or "failed")
        # Trust the agent's safety only if it agrees with our classifier; the
        # classifier is authoritative (an agent must not relabel an unsafe fix).
        # A "resume_monitoring" verdict (the run was never broken — the agent
        # only extended the monitor deadline and is re-watching the SAME live
        # process) is non-destructive by construction, so it is SAFE even though
        # it carries no fix_type from the SAFE menu. Without this it would be
        # classified unsafe and wrongly stop the loop for user confirmation.
        safety = "safe" if outcome == "resume_monitoring" else classify_fix_safety(fix_type)
        return cls(
            attempt=attempt,
            failure_category=str(verdict.get("failure_category") or "unknown"),
            diagnosis=str(verdict.get("diagnosis") or ""),
            fix_type=fix_type,
            fix_safety=safety,
            fix_applied=str(verdict.get("fix_applied") or ""),
            canary_rerun=bool(verdict.get("canary_rerun")),
            relaunched=bool(verdict.get("relaunched")),
            new_pid=_coerce_int(verdict.get("new_pid")),
            outcome=outcome,
            evidence=str(verdict.get("evidence") or ""),
        )


def _coerce_int(v) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


@dataclass
class RecoveryLedger:
    """All recovery attempts for a run + the final outcome."""

    attempts: list[RecoveryAttempt] = field(default_factory=list)
    final_outcome: str = "not_attempted"  # recovered | exhausted | needs_user | not_attempted

    @property
    def recovered(self) -> bool:
        return self.final_outcome == "recovered"

    def to_dict(self) -> dict:
        return {
            "attempts": [asdict(a) for a in self.attempts],
            "final_outcome": self.final_outcome,
            "n_attempts": len(self.attempts),
        }

    def write_json(self, path) -> None:
        from pathlib import Path

        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def build_recovery_context(
    *,
    attempt: int,
    max_attempts: int,
    failure: dict | None,
    config: dict | None,
    last_checkpoint: str | None,
    canary_passed: bool,
    log_tail: str,
    ssh_alias: str,
    remote_run_dir: str,
    local_run_dir: str,
) -> str:
    """Render the structured diagnosis input for the recovery agent."""
    failure = failure or {}
    config = config or {}
    return (
        f"recovery_attempt = {attempt} of {max_attempts}\n"
        f"canary_passed = {canary_passed}\n"
        f"failure_category = {failure.get('failure_category', 'unknown')}\n"
        f"failure_summary = {failure.get('summary', '')}\n"
        f"failure_remediation = {failure.get('remediation', '')}\n"
        f"missing_packages = {failure.get('missing_packages', [])}\n"
        f"failing_file = {failure.get('failing_file', '')}\n"
        f"last_checkpoint = {last_checkpoint or 'none'}\n"
        f"ssh_alias = {ssh_alias}\n"
        f"remote_run_dir = {remote_run_dir}\n"
        f"local_run_dir = {local_run_dir}\n"
        f"effective_config = {json.dumps(config)}\n"
        f"--- last training log (tail) ---\n{log_tail}\n"
        f"--- safe fix types (auto-apply) ---\n{sorted(SAFE_FIX_TYPES)}\n"
        f"--- unsafe fix types (stop + ask user) ---\n{sorted(UNSAFE_FIX_TYPES)}\n"
    )


def render_recovery_report_markdown(ledger: RecoveryLedger) -> str:
    """Render the ## Training Recovery section for RUN_REPORT.md."""
    rows = [
        {
            "attempt": a.attempt,
            "failure": a.failure_category,
            "diagnosis": (a.diagnosis[:80] or EMDASH),
            "fix": (a.fix_applied[:80] or a.fix_type or EMDASH),
            "safety": a.fix_safety,
            "canary": "yes" if a.canary_rerun else "no",
            "outcome": a.outcome,
        }
        for a in ledger.attempts
    ]
    table = markdown_table(
        rows,
        columns=["attempt", "failure", "diagnosis", "fix", "safety", "canary", "outcome"],
        headers={
            "attempt": "Attempt",
            "failure": "Failure",
            "diagnosis": "Diagnosis",
            "fix": "Fix Applied",
            "safety": "Safety",
            "canary": "Canary Rerun",
            "outcome": "Outcome",
        },
        align={"attempt": "right"},
    )
    verdict = {
        "recovered": "Training recovered after agentic diagnosis.",
        "exhausted": "Recovery exhausted all attempts without success — see the failure report.",
        "needs_user": "Recovery proposed an UNSAFE fix and stopped for user confirmation.",
        "not_attempted": "No post-canary recovery was attempted.",
    }.get(ledger.final_outcome, ledger.final_outcome)
    return "\n".join(["## Training Recovery", "", table, "", f"_Outcome:_ {verdict}"])

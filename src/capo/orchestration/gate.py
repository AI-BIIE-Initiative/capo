"""
CAPO 3-step gate as a programmatic state machine.

Orchestrator agent calls into this module rather than performing the
gate logic itself, which:
  1. Stops the agent from re-deriving the same routing on every retry
  2. Makes the gate testable in isolation
  3. Keeps the user-bounce-back path explicit instead of buried in a prompt

Step 1 — script + schema check     pure Python, no LLM call
Step 2 — feasibility probe         remote SSH + probe_result.json parsing
Step 3 — cost gate                 ĉ vs c_max vs α·c_max

On script_bug failures (Step 1 or Step 2), the repair ladder fires:
  Attempt 1 + 2  same orchestrator, compact packet                cheap
  Attempt 3      code-repair-critic subagent, compact packet      rare

If the ladder exhausts, the gate replaces the candidate (next from
model_selection.json top-3); if no candidates remain, it pauses for the user.

User bounce-back triggers:
  schema_user_only_info        — Step 1 missing info only user can fill
  probe_data_schema_user_only  — Step 2 data_schema_mismatch beyond re-profile
  cost_accept_overrun          — Step 3 c_max < ĉ ≤ α·c_max

Each writes <local_run_dir>/reports/pending_question.json and returns a
PauseRequest result; the caller flips SessionState.paused = True and exits.
capo_resume.py replays the question and re-enters at the same step.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from capo.orchestration.probe_failure_packet import (
    build_compact_packet,
    validate_diff,
    write_packet,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class GateOutcome(str, Enum):
    """Top-level result of a single gate.run() call."""
    LAUNCH = "launch"                     # all three steps passed -> proceed to Phase 4
    PAUSE = "pause"                       # awaiting user input — state.paused = True
    REJECT = "reject"                     # no candidates left; report and stop
    FAILED = "failed"                     # unrecoverable structural failure


class StepResult(str, Enum):
    PASS = "pass"
    FAIL_SCRIPT_BUG = "fail_script_bug"
    FAIL_SCHEMA_REPROFILEABLE = "fail_schema_reprofileable"
    FAIL_SCHEMA_USER_ONLY = "fail_schema_user_only"
    FAIL_OOM = "fail_oom"
    FAIL_NAN_INF = "fail_nan_inf"
    FAIL_RESOURCE_MISMATCH = "fail_resource_mismatch"
    FAIL_DATA_SCHEMA_MISMATCH = "fail_data_schema_mismatch"
    EXHAUSTED = "exhausted"


@dataclass
class GateResult:
    outcome: GateOutcome
    candidate_index: int                  # which model_selection candidate ran
    step_reached: int                     # 1, 2, or 3
    narration: str                        # short summary for the orchestrator agent
    pending_question_path: str | None = None
    pause_reason: str = ""
    pause_context: dict[str, Any] = field(default_factory=dict)
    failure_packet_path: str | None = None


# ---------------------------------------------------------------------------
# Callback contracts
# ---------------------------------------------------------------------------

# Mechanical fix function signature — modifies the recipe file in place,
# returns a one-line summary of what it changed.
MechanicalFixFn = Callable[[Path], str]

# Self-repair: orchestrator turn with the compact packet only. Returns True
# if the repair was applied (regardless of whether it later passes the probe).
SelfRepairFn = Callable[[dict[str, Any]], bool]

# Critic: code-repair-critic subagent. Receives the packet path, returns
# the path to the emitted diff file (or None if the critic gave up).
CriticInvokeFn = Callable[[Path], Optional[Path]]

# Probe runner: executes the remote probe, returns the parsed probe_result.json
# as a dict. Raises on infrastructure failures.
ProbeRunnerFn = Callable[[], dict[str, Any]]

# Profiler re-run: invoked when data_schema_mismatch may be re-profileable.
# Returns True if the re-profile updated profile.json with the missing info.
ProfilerRerunFn = Callable[[str], bool]


# ---------------------------------------------------------------------------
# ThreeStepGate
# ---------------------------------------------------------------------------

class ThreeStepGate:
    """The 3-step gate state machine.

    Construct once per run; call run() once per candidate. On a PAUSE outcome,
    the caller writes SessionState and exits; resume re-instantiates and calls
    run() again with the same candidate_index.

    The gate is callback-driven so that the agent-side concerns (issuing remote
    SSH, invoking subagents) stay in the orchestrator. This keeps gate.py pure
    Python and unit-testable.
    """

    def __init__(
        self,
        *,
        local_run_dir: Path,
        max_cost_usd: float,
        tolerance_threshold: float,
        max_self_repair_attempts: int = 2,
        emit: Callable[[str], None] | None = None,
    ) -> None:
        if not 0.0 <= float(tolerance_threshold) <= 1.0:
            raise ValueError(
                f"tolerance_threshold must be in [0, 1], got {tolerance_threshold!r}"
            )
        self.local_run_dir = Path(local_run_dir)
        self.max_cost_usd = float(max_cost_usd)
        self.tolerance_threshold = float(tolerance_threshold)
        self.max_self_repair_attempts = int(max_self_repair_attempts)
        self._emit = emit or (lambda _msg: None)

    @property
    def cost_overrun_factor(self) -> float:
        """α used by Step 3, derived from tolerance_threshold.

        tolerance=0   → α=1.0 (no overrun band; any over-budget projection
                              immediately replaces the candidate).
        tolerance=0.1 → α=1.1 (projection up to 110 % of budget prompts the user).
        """
        return 1.0 + self.tolerance_threshold

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        candidate_index: int,
        n_candidates: int,
        probe_runner: ProbeRunnerFn,
        self_repair: SelfRepairFn,
        critic_invoke: CriticInvokeFn,
        profiler_rerun: ProfilerRerunFn,
        mechanical_fixes: dict[str, MechanicalFixFn],
        expected_schema: dict[str, Any],
        budget: dict[str, Any],
    ) -> GateResult:
        """Execute the 3-step gate for the current candidate.

        Returns a GateResult describing what happened. The caller is
        responsible for acting on it (writing SessionState, advancing the
        candidate index on REJECT/REPLACE, launching Phase 4 on LAUNCH).
        """
        self._emit_step(1, candidate_index, "starting script + schema check")

        # --- Step 1 ---------------------------------------------------
        s1 = self._step1_script_schema_check()
        if s1 == StepResult.FAIL_SCHEMA_USER_ONLY:
            return self._pause_for_user(
                step=1, candidate_index=candidate_index,
                reason="schema_user_only_info",
                question=self._schema_user_question(),
            )
        if s1 == StepResult.FAIL_SCHEMA_REPROFILEABLE:
            if profiler_rerun("step1_schema_reprofile"):
                s1 = self._step1_script_schema_check()  # re-check after re-profile
        if s1 == StepResult.FAIL_SCRIPT_BUG:
            ladder = self._repair_ladder(
                failing_file=self._infer_failing_file_step1(),
                failure_category="script_bug",
                traceback=self._read_schema_check_log(),
                expected_schema=expected_schema,
                budget=budget,
                rerun=self._step1_script_schema_check,
                self_repair=self_repair,
                critic_invoke=critic_invoke,
            )
            if ladder is StepResult.EXHAUSTED:
                return self._maybe_replace(
                    candidate_index, n_candidates, step=1,
                    reason="step1_script_bug_exhausted",
                )
            s1 = ladder

        if s1 != StepResult.PASS:
            return GateResult(
                outcome=GateOutcome.FAILED,
                candidate_index=candidate_index, step_reached=1,
                narration=f"Step 1 unresolvable: {s1.value}",
            )
        self._emit_step(1, candidate_index, "PASS")

        # --- Step 2 ---------------------------------------------------
        self._emit_step(2, candidate_index, "running feasibility probe")
        probe = probe_runner()
        s2 = self._classify_probe(probe)

        while s2 in (
            StepResult.FAIL_OOM,
            StepResult.FAIL_NAN_INF,
            StepResult.FAIL_RESOURCE_MISMATCH,
        ):
            fix_key = {
                StepResult.FAIL_OOM: "oom",
                StepResult.FAIL_NAN_INF: "nan_inf",
                StepResult.FAIL_RESOURCE_MISMATCH: "resource_mismatch",
            }[s2]
            fn = mechanical_fixes.get(fix_key)
            if fn is None:
                return GateResult(
                    outcome=GateOutcome.FAILED,
                    candidate_index=candidate_index, step_reached=2,
                    narration=f"Step 2 {fix_key}: no mechanical fix wired",
                )
            summary = fn(self.local_run_dir / "profile" / "probe_batch_recipe.json")
            self._emit_step(2, candidate_index, f"mechanical fix ({fix_key}): {summary}")
            probe = probe_runner()
            s2 = self._classify_probe(probe)

            # OOM has a special rule today: two consecutive OOMs → reject.
            if fix_key == "oom" and s2 == StepResult.FAIL_OOM:
                return self._maybe_replace(
                    candidate_index, n_candidates, step=2,
                    reason="step2_double_oom_abort_too_large",
                )

        if s2 == StepResult.FAIL_DATA_SCHEMA_MISMATCH:
            if profiler_rerun("step2_schema_reprofile"):
                probe = probe_runner()
                s2 = self._classify_probe(probe)
            if s2 == StepResult.FAIL_DATA_SCHEMA_MISMATCH:
                return self._pause_for_user(
                    step=2, candidate_index=candidate_index,
                    reason="probe_data_schema_user_only",
                    question=self._probe_schema_user_question(probe),
                )

        if s2 == StepResult.FAIL_SCRIPT_BUG:
            ladder = self._repair_ladder(
                failing_file=probe.get("failing_file") or "probe.py",
                failure_category="script_bug",
                traceback=probe.get("traceback") or probe.get("error_message", ""),
                expected_schema=expected_schema,
                budget=budget,
                rerun=lambda: self._classify_probe(probe_runner()),
                self_repair=self_repair,
                critic_invoke=critic_invoke,
            )
            if ladder is StepResult.EXHAUSTED:
                return self._maybe_replace(
                    candidate_index, n_candidates, step=2,
                    reason="step2_script_bug_exhausted",
                )
            s2 = ladder

        if s2 != StepResult.PASS:
            return GateResult(
                outcome=GateOutcome.FAILED,
                candidate_index=candidate_index, step_reached=2,
                narration=f"Step 2 unresolvable: {s2.value}",
            )
        self._emit_step(2, candidate_index, "PASS")

        # --- Step 3 ---------------------------------------------------
        self._emit_step(3, candidate_index, "computing cost gate")
        c_probe, c_max, alpha = self._compute_costs(probe)
        if not math.isfinite(c_probe):
            return GateResult(
                outcome=GateOutcome.FAILED,
                candidate_index=candidate_index, step_reached=3,
                narration="Step 3 failed: projected cost is not finite",
            )

        if c_probe <= c_max:
            self._emit_step(3, candidate_index, f"LAUNCH (ĉ=${c_probe:.2f} ≤ c_max=${c_max:.2f})")
            return GateResult(
                outcome=GateOutcome.LAUNCH,
                candidate_index=candidate_index, step_reached=3,
                narration=(
                    f"Gate passed for candidate[{candidate_index}]: ĉ=${c_probe:.2f}, "
                    f"c_max=${c_max:.2f}. Proceeding to Phase 4 canary launch."
                ),
            )

        if c_probe <= alpha * c_max:
            return self._pause_for_user(
                step=3, candidate_index=candidate_index,
                reason="cost_accept_overrun",
                question=self._cost_overrun_question(c_probe, c_max, alpha),
            )

        # c_probe > α · c_max — replace or reject.
        return self._maybe_replace(
            candidate_index, n_candidates, step=3,
            reason="step3_cost_above_alpha_cmax",
            extra_context={"c_probe": c_probe, "c_max": c_max, "alpha": alpha},
        )

    # ------------------------------------------------------------------
    # Step 1
    # ------------------------------------------------------------------

    def _step1_script_schema_check(self) -> StepResult:
        """Validate that scripts exist on disk and that profile.json carries the
        info needed by the model+strategy contract.

        This is pure local Python. It must NOT issue any LLM call or remote SSH.
        """
        # Required entry-point scripts, package layout, and configs.
        # Mirrors the canonical layout enforced by capo.utils.checks (preflight)
        # and prepare_remote_run_dir.
        required = [
            "probe.py",
            "train.py",
            "requirements.txt",
            "configs/experiment.yaml",
            "configs/training.yaml",
            "configs/evaluation.yaml",
            "src/__init__.py",
            "src/data/__init__.py",
            "src/models/__init__.py",
            "src/train/__init__.py",
            "src/eval/__init__.py",
            "src/utils/__init__.py",
        ]
        missing = [p for p in required if not (self.local_run_dir / p).exists()]
        if missing:
            self._write_schema_check_log(
                "Missing required files at Step 1 script-schema check:\n  "
                + "\n  ".join(missing)
            )
            return StepResult.FAIL_SCRIPT_BUG

        # Forbidden top-level subdirs — the agent must not stash code under a
        # nested experiment dir. capo.utils.checks knows the full set; gate
        # Step 1 enforces the most common offenders explicitly so the failure
        # narration is direct.
        forbidden_dirs = ("fine-tuning", "finetuning", "training", "ft")
        bad = [d for d in forbidden_dirs if (self.local_run_dir / d).is_dir()]
        if bad:
            self._write_schema_check_log(
                "Forbidden subdirectories detected at run root:\n  "
                + "\n  ".join(bad)
                + "\nMove their contents into src/ + configs/ + scripts/ and "
                "delete the forbidden dir before re-running."
            )
            return StepResult.FAIL_SCRIPT_BUG

        # Forbidden Python files at run root — only probe.py and train.py allowed.
        stray_py = [
            p.name
            for p in self.local_run_dir.iterdir()
            if p.is_file() and p.suffix == ".py" and p.name not in {"probe.py", "train.py"}
        ]
        if stray_py:
            self._write_schema_check_log(
                "Stray Python files at run root (move into src/<area>/):\n  "
                + "\n  ".join(stray_py)
            )
            return StepResult.FAIL_SCRIPT_BUG

        # profile.json must exist and carry the schema fields downstream needs.
        profile_path = self.local_run_dir / "profile" / "profile.json"
        if not profile_path.exists():
            profile_path = self.local_run_dir / "profile.json"
        if not profile_path.exists():
            self._write_schema_check_log("profile.json missing")
            return StepResult.FAIL_SCHEMA_REPROFILEABLE

        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._write_schema_check_log(f"profile.json unreadable: {exc}")
            return StepResult.FAIL_SCHEMA_REPROFILEABLE

        # Task type is user-only knowledge when profile cannot infer it
        # (regression vs classification can be ambiguous from a numeric column).
        task_type = profile.get("task_type") or profile.get("recommended_task_type")
        if not task_type:
            return StepResult.FAIL_SCHEMA_USER_ONLY

        # Label column: if the profile lists candidates but no chosen column,
        # that is also user-only territory.
        if "label_column" not in profile and "target_column" not in profile:
            candidates = profile.get("label_column_candidates") or []
            if len(candidates) > 1:
                return StepResult.FAIL_SCHEMA_USER_ONLY
            if len(candidates) == 0:
                return StepResult.FAIL_SCHEMA_REPROFILEABLE

        return StepResult.PASS

    def _schema_user_question(self) -> dict[str, Any]:
        """Build the AskUserQuestion payload for schema_user_only_info."""
        profile_path = self.local_run_dir / "profile" / "profile.json"
        if not profile_path.exists():
            profile_path = self.local_run_dir / "profile.json"
        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            profile = {}
        candidates = profile.get("label_column_candidates") or []
        if not profile.get("task_type") and not profile.get("recommended_task_type"):
            return {
                "header": "Task type",
                "question": (
                    "The profiler could not determine whether this dataset is a "
                    "regression or classification task. Which fits?"
                ),
                "options": [
                    {"label": "classification", "description": "Discrete labels / classes"},
                    {"label": "regression", "description": "Continuous numerical target"},
                    {"label": "language model", "description": "Sequence likelihood / perplexity"},
                ],
                "answer_target": "profile.task_type",
            }
        return {
            "header": "Label column",
            "question": (
                "The profiler found multiple plausible label columns. Which one is the "
                "training target?"
            ),
            "options": [{"label": c, "description": ""} for c in candidates[:4]],
            "answer_target": "profile.label_column",
        }

    def _write_schema_check_log(self, msg: str) -> None:
        log = self.local_run_dir / "reports" / "schema_check.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(msg, encoding="utf-8")

    def _read_schema_check_log(self) -> str:
        log = self.local_run_dir / "reports" / "schema_check.log"
        if not log.exists():
            return ""
        return log.read_text(encoding="utf-8")

    def _infer_failing_file_step1(self) -> str:
        """Best guess at which file Step 1's script_bug refers to.

        For now this returns the most likely culprit (probe entry-point first);
        the compact packet includes schema_check.log so the repair turn sees
        the full picture.
        """
        return "probe.py"

    # ------------------------------------------------------------------
    # Step 2 — probe classification
    # ------------------------------------------------------------------

    def _classify_probe(self, probe: dict[str, Any]) -> StepResult:
        if probe.get("success") is True:
            return StepResult.PASS
        category = probe.get("failure_category")
        mapping = {
            "oom": StepResult.FAIL_OOM,
            "nan_inf": StepResult.FAIL_NAN_INF,
            "resource_mismatch": StepResult.FAIL_RESOURCE_MISMATCH,
            "data_schema_mismatch": StepResult.FAIL_DATA_SCHEMA_MISMATCH,
            "script_bug": StepResult.FAIL_SCRIPT_BUG,
        }
        return mapping.get(category, StepResult.FAIL_SCRIPT_BUG)

    def _probe_schema_user_question(self, probe: dict[str, Any]) -> dict[str, Any]:
        return {
            "header": "Schema gap",
            "question": (
                "The feasibility probe failed with data_schema_mismatch and a stricter "
                "re-profile did not resolve it. The dataset is missing information only "
                "you can provide. Please describe the label semantics (or the missing "
                "field) below."
            ),
            "options": [
                {"label": "Provide details", "description": "Free-text answer"},
            ],
            "answer_target": "profile.label_semantics",
            "probe_error": probe.get("error_message", ""),
        }

    # ------------------------------------------------------------------
    # Step 3 — cost gate
    # ------------------------------------------------------------------

    def _compute_costs(self, probe: dict[str, Any]) -> tuple[float, float, float]:
        """Return (ĉ_probe, c_max, α). Reads cost_report.json if present;
        falls back to recomputing from probe + pricing if not.
        """
        c_max = self.max_cost_usd
        alpha = self.cost_overrun_factor

        report_path = self.local_run_dir / "pricing" / "cost_report.json"
        if not report_path.exists():
            # Backward-compat: older runs wrote cost_report.json at run root.
            legacy = self.local_run_dir / "cost_report.json"
            if legacy.exists():
                report_path = legacy
        if report_path.exists():
            try:
                rep = json.loads(report_path.read_text(encoding="utf-8"))
                c_probe = float(rep.get("projected_cost_usd", float("inf")))
                return c_probe, c_max, alpha
            except (OSError, json.JSONDecodeError, ValueError):
                pass

        # No cost_report.json yet — the orchestrator hasn't synthesised it.
        # The gate cannot compute it without epochs and n_samples; treat as +inf.
        return float("inf"), c_max, alpha

    def _cost_overrun_question(self, c_probe: float, c_max: float, alpha: float) -> dict[str, Any]:
        return {
            "header": "Accept overrun?",
            "question": (
                f"Projected cost ${c_probe:.2f} exceeds your budget ${c_max:.2f} "
                f"but stays under the {alpha:.1f}× threshold (${alpha * c_max:.2f}). "
                "Accept the overrun and launch?"
            ),
            "options": [
                {"label": "accept", "description": f"Launch at projected ${c_probe:.2f}"},
                {"label": "reject", "description": "Replace with next candidate or abort"},
            ],
            "answer_target": "cost.accept_overrun",
            "c_probe": c_probe,
            "c_max": c_max,
            "alpha": alpha,
        }

    # ------------------------------------------------------------------
    # Repair ladder
    # ------------------------------------------------------------------

    def _repair_ladder(
        self,
        *,
        failing_file: str,
        failure_category: str,
        traceback: str,
        expected_schema: dict[str, Any],
        budget: dict[str, Any],
        rerun: Callable[[], StepResult],
        self_repair: SelfRepairFn,
        critic_invoke: CriticInvokeFn,
    ) -> StepResult:
        """Run the bounded repair ladder for a script_bug failure.

        Returns the StepResult of the most recent re-check, or EXHAUSTED if
        all attempts ran without resolving.
        """
        history: list[dict[str, Any]] = []

        # Attempts 1, 2 — same orchestrator, compact packet
        for attempt in range(1, self.max_self_repair_attempts + 1):
            packet = build_compact_packet(
                failing_file=failing_file,
                failure_category=failure_category,
                traceback=traceback,
                expected_schema=expected_schema,
                budget=budget,
                history=history,
                failing_file_root=self.local_run_dir,
            )
            packet_path = self._write_packet_for_attempt(packet, attempt)
            self._emit_step(0, 0, f"repair Attempt {attempt}: orchestrator self-repair")
            applied = self_repair(packet)
            history.append({
                "attempt": attempt,
                "summary": "orchestrator self-repair" if applied else "self-repair returned no change",
                "diff_path": str(packet_path),
            })
            outcome = rerun()
            if outcome == StepResult.PASS:
                self._emit_step(0, 0, f"repair Attempt {attempt}: PASS")
                return StepResult.PASS

        # Attempt 3 — code-repair-critic subagent
        packet = build_compact_packet(
            failing_file=failing_file,
            failure_category=failure_category,
            traceback=traceback,
            expected_schema=expected_schema,
            budget=budget,
            history=history,
            failing_file_root=self.local_run_dir,
        )
        packet_path = self._write_packet_for_attempt(packet, 3)
        self._emit_step(0, 0, "repair Attempt 3: invoking code-repair-critic")
        diff_path = critic_invoke(packet_path)
        if diff_path is None:
            self._emit_step(0, 0, "repair Attempt 3: critic returned INSUFFICIENT_INFO")
            return StepResult.EXHAUSTED

        diff_text = diff_path.read_text(encoding="utf-8") if diff_path.exists() else ""
        if not validate_diff(diff_text):
            self._emit_step(0, 0, "repair Attempt 3: critic diff did not validate")
            return StepResult.EXHAUSTED

        # Application of the diff happens in the orchestrator (it owns git apply).
        # We just signal that the critic produced a valid diff; the caller's
        # rerun() will exercise the patched file.
        outcome = rerun()
        if outcome == StepResult.PASS:
            self._emit_step(0, 0, "repair Attempt 3: PASS after critic patch")
            return StepResult.PASS

        self._emit_step(0, 0, "repair Attempt 3: critic patch did not fix the failure")
        return StepResult.EXHAUSTED

    def _write_packet_for_attempt(self, packet: dict[str, Any], attempt: int) -> Path:
        out = self.local_run_dir / "reports" / "repairs" / f"attempt_{attempt}_packet.json"
        return write_packet(packet, out)

    # ------------------------------------------------------------------
    # Candidate replacement / rejection
    # ------------------------------------------------------------------

    def _maybe_replace(
        self,
        candidate_index: int,
        n_candidates: int,
        *,
        step: int,
        reason: str,
        extra_context: dict[str, Any] | None = None,
    ) -> GateResult:
        """Decide between REPLACE (advance candidate) and REJECT (no more candidates)."""
        if candidate_index + 1 < n_candidates:
            return GateResult(
                outcome=GateOutcome.PAUSE,   # treat replacement as a "pause-and-restart"
                candidate_index=candidate_index, step_reached=step,
                narration=(
                    f"Step {step} exhausted on candidate[{candidate_index}] "
                    f"({reason}). Replace with candidate[{candidate_index + 1}] "
                    "and re-enter the gate."
                ),
                pause_reason=f"replace_candidate:{reason}",
                pause_context={
                    "next_candidate_index": candidate_index + 1,
                    "exhausted_step": step,
                    "reason": reason,
                    **(extra_context or {}),
                },
            )
        return GateResult(
            outcome=GateOutcome.REJECT,
            candidate_index=candidate_index, step_reached=step,
            narration=(
                f"Step {step} exhausted on the last candidate ({reason}). "
                "No more candidates available — rejecting the run."
            ),
            pause_reason=reason,
            pause_context={"exhausted_step": step, **(extra_context or {})},
        )

    # ------------------------------------------------------------------
    # Pause helper (user bounce-back)
    # ------------------------------------------------------------------

    def _pause_for_user(
        self,
        *,
        step: int,
        candidate_index: int,
        reason: str,
        question: dict[str, Any],
    ) -> GateResult:
        out = self.local_run_dir / "reports" / "pending_question.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(question, indent=2), encoding="utf-8")
        narration = (
            f"Step {step} paused for user input ({reason}). Pending question "
            f"at {out.relative_to(self.local_run_dir)}. Run capo_resume after "
            "the user answers."
        )
        self._emit_step(step, candidate_index, narration)
        return GateResult(
            outcome=GateOutcome.PAUSE,
            candidate_index=candidate_index, step_reached=step,
            narration=narration,
            pending_question_path=str(out.relative_to(self.local_run_dir)),
            pause_reason=reason,
            pause_context={"step": step, "candidate_index": candidate_index},
        )

    # ------------------------------------------------------------------
    # Narration helper
    # ------------------------------------------------------------------

    def _emit_step(self, step: int, candidate_index: int, msg: str) -> None:
        if step == 0:
            self._emit(f"[gate] {msg}")
        else:
            self._emit(f"[gate] step {step} candidate[{candidate_index}]: {msg}")


# ---------------------------------------------------------------------------
# Mechanical-fix helpers (used by the orchestrator to populate mechanical_fixes)
# ---------------------------------------------------------------------------

def mechanical_fix_oom(recipe_path: Path) -> str:
    """Halve probe_batch_size in probe_batch_recipe.json. Returns a summary."""
    recipe = json.loads(recipe_path.read_text(encoding="utf-8")) if recipe_path.exists() else {}
    old = int(recipe.get("probe_batch_size", 8))
    new = max(1, old // 2)
    recipe["probe_batch_size"] = new
    recipe_path.write_text(json.dumps(recipe, indent=2), encoding="utf-8")
    return f"probe_batch_size {old} → {new}"


def mechanical_fix_nan_inf(recipe_path: Path) -> str:
    """Enable bf16 and drop the learning rate 10×."""
    recipe = json.loads(recipe_path.read_text(encoding="utf-8")) if recipe_path.exists() else {}
    recipe["precision"] = "bf16"
    old_lr = float(recipe.get("learning_rate", 1e-4))
    new_lr = old_lr / 10
    recipe["learning_rate"] = new_lr
    recipe_path.write_text(json.dumps(recipe, indent=2), encoding="utf-8")
    return f"precision→bf16, lr {old_lr:.2e} → {new_lr:.2e}"


def mechanical_fix_resource_mismatch(recipe_path: Path) -> str:
    """Halve probe_batch_size and double grad_accum_steps."""
    recipe = json.loads(recipe_path.read_text(encoding="utf-8")) if recipe_path.exists() else {}
    old_b = int(recipe.get("probe_batch_size", 8))
    new_b = max(1, old_b // 2)
    recipe["probe_batch_size"] = new_b
    old_g = int(recipe.get("grad_accum_steps", 1))
    new_g = max(1, old_g * 2)
    recipe["grad_accum_steps"] = new_g
    recipe_path.write_text(json.dumps(recipe, indent=2), encoding="utf-8")
    return f"probe_batch_size {old_b}→{new_b}, grad_accum {old_g}→{new_g}"


DEFAULT_MECHANICAL_FIXES: dict[str, MechanicalFixFn] = {
    "oom": mechanical_fix_oom,
    "nan_inf": mechanical_fix_nan_inf,
    "resource_mismatch": mechanical_fix_resource_mismatch,
}


# ---------------------------------------------------------------------------
# Narration helpers — short structured blobs the orchestrator agent reads
# between steps to keep situational awareness without re-deriving routing.
# ---------------------------------------------------------------------------

def narration_blob(result: GateResult) -> str:
    """Render the gate result as a short structured string for the agent.

    Designed to be ≤200 input tokens. The agent reads this between steps so
    its context stays coherent for Phases 4–6, but it does NOT re-derive any
    routing decision the gate already made.
    """
    payload = {
        "gate_outcome": result.outcome.value,
        "candidate_index": result.candidate_index,
        "step_reached": result.step_reached,
        "pause_reason": result.pause_reason or None,
        "pending_question_path": result.pending_question_path,
    }
    return (
        "GATE RESULT (do not re-derive routing):\n"
        + json.dumps(payload, indent=2)
        + f"\n\nSummary: {result.narration}"
    )


def result_to_dict(result: GateResult) -> dict[str, Any]:
    """Serialise a GateResult to a dict (for state.json or pause_context)."""
    d = asdict(result)
    d["outcome"] = result.outcome.value
    return d

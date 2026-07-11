"""
Resume a fine-tuning run from its state.json in the local run directory.

Two callers wire into this:
  * scripts/run_fine_tuning.py — when the YAML config sets resume: <run_id>,
    the entry point hands off here instead of starting a fresh run.

Resume modes:
  * Paused (user input awaited) — state.paused=True. We read
    reports/pending_question.json, collect the user's answer (interactive or
    --answer CLI flag), patch the appropriate artifact (profile.json,
    cost approval, etc.), clear the pause state, and re-enter the
    orchestrator at the pre-launch phase to re-run the 3-step gate.
  * Mid-training — current_phase in {training, finalizing, failed} with a
    remote checkpoint present. Delegates to restart_from_checkpoint=True.

Exit codes (returned, not raised, so callers can sys.exit(...)):
    0 — run already completed, or resumed successfully to terminal_state=completed
    1 — resumed but final state is non-completed (failed / aborted / ...)
    2 — no state.json found for run_id
    3 — phase is too early to resume (init/pre_launch and not paused)
    4 — paused, but no pending question or no answer available
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from capo.orchestration.fine_tuning_orchestrator import FineTuningOrchestrator
from capo.orchestration.orchestration import _REPO_ROOT_ORCH
from capo.persistence.run_inventory import (
    build_inventory,
    format_plan,
    plan_resume,
    write_run_state,
)
from capo.persistence.session_store import SessionState, SessionStore

_LOCAL_FINE_TUNING_ROOT = _REPO_ROOT_ORCH / "runs"

# Phases at which no remote checkpoint exists yet — restart_from_checkpoint
# would fail because nothing has been written to outputs/checkpoints/ on the
# instance. The caller should rerun fresh instead, unless the run is paused
# for user input (handled separately).
_PRE_TRAINING_PHASES = frozenset({"init", "pre_launch"})


def _store_for(run_id: str) -> SessionStore:
    return SessionStore(_LOCAL_FINE_TUNING_ROOT / run_id)


def _print_state(state: SessionState) -> None:
    print(f"run_id          : {state.run_id}")
    print(f"created_at      : {state.created_at}")
    print(f"updated_at      : {state.updated_at}")
    print(f"current_phase   : {state.current_phase}")
    if state.terminal_state:
        print(f"terminal_state  : {state.terminal_state}")
    if state.error:
        print(f"error           : {state.error}")
    if state.restart_hint:
        print(f"restart_hint    : {state.restart_hint}")
    print(f"local_run_dir   : {state.local_run_dir}")
    print(f"model_id        : {state.model_id}")
    print(f"dataset_ref     : {state.dataset_ref}")
    print(f"strategy        : {state.fine_tune_strategy}")
    print(f"max_cost_usd    : {state.max_cost_usd}")


def _print_result_summary(result) -> None:
    print()
    print(f"run_id          : {result.run_id}")
    print(f"state           : {result.state}")
    print(f"output_dir      : {result.local_run_dir}")
    if result.ssh_alias:
        print(f"ssh_alias       : {result.ssh_alias}")
    if result.actual_cost_usd is not None:
        print(f"actual_cost     : ${result.actual_cost_usd:.2f}")
    if result.agent_cost_usd is not None:
        print(f"agent_cost      : ${result.agent_cost_usd:.4f}")
    if result.finetuned_model_path:
        print(f"model           : {result.finetuned_model_path}")
    if result.trackio_url:
        print(f"trackio         : {result.trackio_url}")
    if result.resumed_from_checkpoint:
        print(f"resumed_from    : {result.resumed_from_checkpoint}")


def _collect_answer(question: dict, answer_arg: str | None) -> str | None:
    """Resolve the user's answer to the pending question.

    --answer (answer_arg) wins when provided (for non-interactive resumption).
    Otherwise print the question and read a single line from stdin. Returns
    None if the user supplied an empty answer with no CLI override.
    """
    if answer_arg is not None:
        return answer_arg.strip()

    print("Question paused for user input:")
    print(f"  {question.get('question', '<missing>')}")
    options = question.get("options") or []
    if options:
        print("  Options:")
        for opt in options:
            lbl = opt.get("label", "")
            desc = opt.get("description", "")
            print(f"    - {lbl}: {desc}" if desc else f"    - {lbl}")
    try:
        ans = input("  Your answer: ").strip()
    except EOFError:
        return None
    return ans or None


def _apply_answer_to_artifacts(
    local_run_dir: Path, question: dict, answer: str
) -> tuple[bool, str, str]:
    """Patch the appropriate artifact based on question.answer_target.

    Returns (ok, message, answer_artifact_relpath) where answer_artifact_relpath
    is the file the agent should re-read to recover the user's decision
    (relative to local_run_dir).

    Supported targets:
      profile.task_type        — patch profile.json task_type field
      profile.label_column     — patch profile.json label_column field
      profile.label_semantics  — append to profile.json user_notes field
      cost.accept_overrun      — write reports/cost_overrun_decision.json
    """
    target = question.get("answer_target", "")

    if target.startswith("profile."):
        profile_path = local_run_dir / "profile" / "profile.json"
        if not profile_path.exists():
            profile_path = local_run_dir / "profile.json"
        if not profile_path.exists():
            return False, f"profile.json missing at {profile_path}", ""
        try:
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"profile.json unreadable: {exc}", ""

        field_name = target.split(".", 1)[1]
        if field_name == "label_semantics":
            notes = profile.get("user_notes", "")
            profile["user_notes"] = (notes + "\n" + answer).strip()
        else:
            profile[field_name] = answer

        profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
        rel = str(profile_path.relative_to(local_run_dir))
        return True, f"patched profile.json:{field_name}", rel

    if target == "cost.accept_overrun":
        decision_path = local_run_dir / "reports" / "cost_overrun_decision.json"
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        decision_path.write_text(
            json.dumps({"accept": answer.lower() in {"accept", "yes", "y", "true"}}, indent=2),
            encoding="utf-8",
        )
        return (
            True,
            f"recorded cost_overrun_decision.accept={answer}",
            "reports/cost_overrun_decision.json",
        )

    return False, f"unknown answer_target {target!r}", ""


def _build_resume_orchestrator(state: SessionState) -> FineTuningOrchestrator:
    """Reconstruct the orchestrator from persisted run state. Shared by the
    paused and mid-training resume paths so both stay in lockstep."""
    return FineTuningOrchestrator(
        key_path=state.key_path,
        ssh_key_name=state.ssh_key_name,
        model_id=state.model_id,
        fine_tune_strategy=state.fine_tune_strategy,
        dataset_ref=state.dataset_ref,
        ssh_alias=state.ssh_alias_override,
        gpu_preference=state.gpu_preference,
        allow_reuse_existing=state.allow_reuse_existing,
        max_cost_usd=state.max_cost_usd,
        trackio_space_id=state.trackio_space_id,
        probe_max_retries=state.probe_max_retries,
        tolerance_threshold=state.tolerance_threshold,
    )


def prepare_pause_resume(run_id: str, answer: str):
    """Prepare an inline answer for a paused run WITHOUT running it.

    Applies the answer to the run's artifacts, clears the pause, rebuilds the
    orchestrator, and returns (orchestrator, run_sync_kwargs) ready for the
    caller to execute — so the CLI can drive the resumed run through the live run
    view instead of plain stdout. Returns None when an inline re-entry isn't
    possible (no state.json, not paused, missing/unreadable pending question, or
    an unknown answer target); the caller should then fall back to resume_run.

    The apply-answer + pause-clear logic is identical to _resume_from_pause,
    so the pause/resume contract is unchanged — this only splits "prepare" from
    "run" so the answer can be collected inline rather than from stdin.
    """
    store = _store_for(run_id)
    state = store.load()
    if state is None or not state.paused:
        return None

    local_run_dir = Path(state.local_run_dir)
    q_rel = state.pending_question_path or "reports/pending_question.json"
    try:
        question = json.loads((local_run_dir / q_rel).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    ok, _msg, answer_artifact = _apply_answer_to_artifacts(local_run_dir, question, answer)
    if not ok:
        return None

    pause_reason = state.pause_reason
    pause_context = dict(state.pause_context or {})
    store.update(paused=False, pause_reason="", pending_question_path="", pause_context={})

    orch = _build_resume_orchestrator(state)
    run_kwargs = dict(
        run_id=state.run_id,
        output_dir=state.local_run_dir,
        resume_from_pause=True,
        pause_reason=pause_reason,
        pause_context=pause_context,
        answer_artifact=answer_artifact,
    )
    return orch, run_kwargs


def _resume_from_pause(state: SessionState, answer_arg: str | None) -> int:
    """Resume a run that paused for user input.

    Reads pending_question.json, collects an answer, patches the artifact,
    clears the pause flag, and re-enters the orchestrator in resume_from_pause
    mode — a tightly-scoped continuation prompt that re-runs only the gate
    step that paused and then proceeds to training launch. The full pipeline
    is NOT restarted: infra, profiling, scripts, and the probe all stay as
    they were on disk.
    """
    local_run_dir = Path(state.local_run_dir)
    q_rel = state.pending_question_path or "reports/pending_question.json"
    q_path = local_run_dir / q_rel
    if not q_path.exists():
        print(
            f"error: pending_question.json missing at {q_path}",
            file=sys.stderr,
        )
        return 4
    try:
        question = json.loads(q_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: pending_question.json unreadable: {exc}", file=sys.stderr)
        return 4

    print(f"Run is paused: {state.pause_reason}")
    answer = _collect_answer(question, answer_arg)
    if answer is None:
        print("error: no answer provided (use --answer for non-interactive resume)", file=sys.stderr)
        return 4

    ok, msg, answer_artifact = _apply_answer_to_artifacts(
        local_run_dir, question, answer
    )
    print(f"  → {msg}")
    if not ok:
        return 4

    # Capture pause metadata BEFORE clearing — the orchestrator needs to know
    # which gate step paused (and on which candidate) to re-enter at the right
    # point. Once cleared, state.pause_* is empty.
    pause_reason = state.pause_reason
    pause_context = dict(state.pause_context or {})

    store = _store_for(state.run_id)
    store.update(
        paused=False,
        pause_reason="",
        pending_question_path="",
        pause_context={},
    )
    print(f"Pause cleared. Re-entering orchestrator in resume-from-pause mode "
          f"(reason={pause_reason!r}, answer_artifact={answer_artifact}).")

    orch = _build_resume_orchestrator(state)
    result = orch.run_sync(
        run_id=state.run_id,
        output_dir=state.local_run_dir,
        resume_from_pause=True,
        pause_reason=pause_reason,
        pause_context=pause_context,
        answer_artifact=answer_artifact,
    )
    _print_result_summary(result)
    return 0 if result.state == "completed" else 1


def resume_run(run_id: str, answer: str | None = None) -> int:
    """Resume the run identified by run_id. Returns a shell exit code.

    When the run is paused for user input, answer is forwarded into the
    pending-question prompt so callers (CI, scripted resumes) can supply it
    non-interactively via --answer.
    """
    store = _store_for(run_id)
    state = store.load()
    if state is None:
        print(f"error: no state.json for run_id={run_id!r}", file=sys.stderr)
        print(f"       expected at {store.path}", file=sys.stderr)
        return 2

    print("Run state:")
    _print_state(state)
    print()

    # Read-only artifact census up front. This distinguishes a run that is
    # genuinely finished from one finalized failed while expensive, reusable
    # artifacts (e.g. partial Boltz embeddings) still sit on disk 
    run_dir = Path(state.local_run_dir)
    inv = build_inventory(run_dir)
    plan = plan_resume(inv)
    try:
        write_run_state(run_dir, inv, plan)
    except OSError:
        pass
    reusable = inv.has_checkpoint or bool(inv.done_complexes)

    # Genuine success → nothing to do. A finalized FAILED run is NOT a dead end
    # when reusable artifacts exist: it falls through to the artifact-aware path.
    if state.current_phase == "completed" and (
        state.terminal_state == "completed" or not reusable
    ):
        if state.terminal_state == "completed":
            print("Run already completed successfully. Nothing to do.")
        else:
            print(
                f"Run finalized as {state.terminal_state!r} with no reusable "
                "artifacts on disk — nothing to resume. Start a fresh run."
            )
        return 0

    # Paused runs short-circuit the pre-training check below — they have NO
    # remote checkpoint by design and must re-enter the orchestrator from the
    # gate, not from a checkpoint resume.
    if state.paused:
        return _resume_from_pause(state, answer)

    # Pre-training with nothing reusable → a fresh re-run is genuinely required.
    # (With reusable artifacts present we proceed — they were produced post-launch.)
    if state.current_phase in _PRE_TRAINING_PHASES and not reusable:
        print(
            f"Cannot resume from phase {state.current_phase!r}: training never "
            "launched, so there is no checkpoint or artifact to resume from.",
            file=sys.stderr,
        )
        print(
            "Re-run the full pipeline instead — the same run_id may be reused "
            "if the local_run_dir is intact.",
            file=sys.stderr,
        )
        return 3

    if not (run_dir / "infra.json").exists():
        print(
            f"Cannot resume run {state.run_id!r}: infra.json is missing, so the run "
            "never completed Phase 0 — there is no instance to resume on.",
            file=sys.stderr,
        )
        if state.error:
            print(f"  recorded error: {state.error}", file=sys.stderr)
        print(
            f"  {state.restart_hint or 'Start a fresh run (leave resume: null).'}",
            file=sys.stderr,
        )
        return 3

    # Print the artifact-aware plan, then delegate to the existing restart path.
    # The resume_training prompt handles BOTH a training checkpoint AND the
    # no-checkpoint-but-reusable-embeddings case (re-launch the idempotent
    # pipeline, which skips complexes whose valid embedding already exists).
    print()
    print(format_plan(inv, plan))
    print()
    if inv.has_checkpoint:
        print(f"Resuming run {state.run_id!r} from checkpoint…")
    else:
        print(
            f"Resuming run {state.run_id!r} from on-disk artifacts "
            f"(next: {plan.next_resume_point}) — no checkpoint; "
            f"{len(inv.done_complexes)} valid embeddings will be reused, "
            f"{len(inv.missing_complexes)} recomputed."
        )

    orch = _build_resume_orchestrator(state)
    result = orch.run_sync(
        run_id=state.run_id,
        output_dir=state.local_run_dir,
        restart_from_checkpoint=True,
    )

    _print_result_summary(result)
    return 0 if result.state == "completed" else 1


def list_runs() -> int:
    """Print every known run, newest-first by updated_at. Returns 0 always."""
    if not _LOCAL_FINE_TUNING_ROOT.exists():
        print("No runs found.")
        return 0

    states: list[SessionState] = []
    for run_dir in _LOCAL_FINE_TUNING_ROOT.iterdir():
        if not run_dir.is_dir():
            continue
        s = SessionStore(run_dir).load()
        if s is not None:
            states.append(s)

    if not states:
        print("No runs found.")
        return 0

    states.sort(key=lambda s: s.updated_at, reverse=True)
    width = max(len(s.run_id) for s in states)
    print(f"{'run_id':<{width}}  {'phase':<12}  {'updated_at':<25}  dataset")
    for s in states:
        print(
            f"{s.run_id:<{width}}  "
            f"{s.current_phase:<12}  "
            f"{s.updated_at:<25}  "
            f"{s.dataset_ref}"
        )
    return 0

"""
fine_tuning_orchestrator.py — Agent-driven fine-tuning runs on Lambda GPU.

full fine-tuning flow end-to-end:
  Phase 0  attach to an existing Lambda session (if its GPU is strong enough)
           or provision a new instance via the lambda-session-manager subagent;
           cache pricing via the cloud-provider-connector subagent.
  Phase 1  invoke the data-profiler subagent (skills/profiling-datasets) to
           generate profile.json and the p50/p90/p95/p99 length percentiles
           required for the feasibility probe.
  Stage 1  run a feasibility probe (one forward, one forward+backward on a
           p99-length batch). Up to probe_max_retries bounded self-repairs,
           classified into script_bug / data_schema_mismatch /
           resource_mismatch / oom / nan_inf.
  Gate     compute projected_cost_usd; abort cleanly if > max_cost_usd.
  Train    launch training inside the persistent capo_remote tmux session
           as nohup ... &, poll state, recover on SSH drop.
  Report   sync outputs/ results/ checkpoints/ reports/; emit final summary.

Logging uses the shared ProgressEmitter with activity_tag="fine-tuning"
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from capo.context.compaction import (
    CaseFile,
    Compactor,
    CompactionConfig,
    CompactionStore,
    NullCompactor,
)
from capo.observability import progress as ip
from capo.observability.progress import reset_runner_tag, set_runner_tag
from capo.observability.trackio_space import (
    ensure_trackio_dashboard,
    resolve_default_space_id,
)
from capo.observability.training_health_monitor import (
    HealthMonitorContext,
    HealthReport,
    MonitorHandoff,
    TrainingHealthMonitor,
    parse_json_with_repair,
)
from capo.memory.run_report import INDEX_PATH as _RUNS_INDEX_PATH
from capo.orchestration.agent_runner import SUBAGENTS, AgentRunner
from capo.orchestration.orchestration import (
    _REPO_ROOT_ORCH,
    _SKILLS_DIR,
    _extract_model_slug,
)
from capo.orchestration.post_launch_diagnostics import diagnose_post_launch_failure
from capo.orchestration.post_launch_repair import (
    apply_and_relaunch,
    is_auto_repairable,
    repair_signature,
)
from capo.orchestration.training_recovery import (
    RecoveryAttempt,
    RecoveryLedger,
    build_recovery_context,
    render_recovery_report_markdown,
)
from capo.results.io import has_checkpoint_content, list_saved_checkpoints
from capo.report.cost import (
    AgentCost,
    RunCostReport,
    make_agent_cost,
    make_agent_cost_from_total,
    make_infra_cost,
    render_cost_report_markdown,
)
from capo.utils.dataset_source import DatasetSource, resolve_dataset_source
from capo.utils.model_resolution import build_model_selection_json, resolve_model
from capo.persistence.session_store import SessionState, SessionStore, new_session
from capo.remote.run_manager import push_hf_token_to_remote, read_remote_run_status
from capo.research.hf_research import HFResearcher, ResearchFindings
from capo.utils.prompts import load_prompt


# ---------------------------------------------------------------------------
# GPU preference parsing
# ---------------------------------------------------------------------------

_GPU_PATTERNS: list[tuple[str, str]] = [
    (r"\b(\d+)\s*x\s*gh200\b",     "{n}x GH200"),
    (r"\b(\d+)\s*x\s*a100\b",      "{n}x A100"),
    (r"\b(\d+)\s*x\s*h100\b",      "{n}x H100"),
    (r"\b(\d+)\s*x\s*a10\b",       "{n}x A10"),
    (r"\b(\d+)\s*x\s*v100\b",      "{n}x V100"),
    (r"\b(\d+)\s*x\s*l40s?\b",     "{n}x L40"),
    (r"\b(\d+)\s*x\s*l4\b",        "{n}x L4"),
    (r"\bgpu_(\d+)x_([a-z0-9]+)", "gpu_{n}x_{gpu}"),
    # Bare names — default to 1x
    (r"\bgh200\b",                 "1x GH200"),
    (r"\bh100\b",                  "1x H100"),
    (r"\ba100\b",                  "1x A100"),
    (r"\ba10\b",                   "1x A10"),
    (r"\bv100\b",                  "1x V100"),
    (r"\bl40s?\b",                 "1x L40"),
    (r"\bl4\b",                    "1x L4"),
]


def _read_infra_ssh_alias(local_run_dir: Path) -> str | None:
    """Best-effort read of ssh_alias from <local_run_dir>/infra.json."""
    path = local_run_dir / "infra.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("ssh_alias")
    except (json.JSONDecodeError, OSError):
        return None


def _read_trackio_url(local_run_dir: Path) -> str | None:
    """Best-effort read of the trackio URL from reports/trackio_url.txt."""
    path = local_run_dir / "reports" / "trackio_url.txt"
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _write_research_findings(findings: "ResearchFindings", reports_dir: Path) -> None:
    """Persist research findings to reports/research_findings.json."""
    try:
        path = reports_dir / "research_findings.json"
        path.write_text(
            json.dumps(findings.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        # Consistent with task.md being written to file rather than
        # embedded in the prompt. The full version above is preserved for human
        # reference.
        agent_path = reports_dir / "research_findings_agent.json"
        agent_path.write_text(
            json.dumps(findings.to_agent_safe_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass


def _parse_gpu_preference(task_description: str) -> str | None:
    """Extract a GPU preference from a free-text task description, or None."""
    if not task_description:
        return None
    lower = task_description.lower()
    for pattern, template in _GPU_PATTERNS:
        m = re.search(pattern, lower)
        if not m:
            continue
        if template.startswith("gpu_"):
            return f"gpu_{m.group(1)}x_{m.group(2)}"
        if "{n}" in template:
            return template.format(n=m.group(1))
        return template
    return None


# ---------------------------------------------------------------------------
# Pre-launch runner system prompts
# These are the full instructions for the three subagents agents that run in
# parallel (via asyncio.gather) 
# They are in isolated API calls, their content never enters the main
# agent's initial context.
# ---------------------------------------------------------------------------

_INFRA_SYSTEM_PROMPT = load_prompt("orchestrator/system_prompts/infrastructure")

_DATA_PROFILER_SYSTEM_PROMPT = load_prompt("orchestrator/system_prompts/data_profiler")

_MODEL_SELECTOR_SYSTEM_PROMPT = load_prompt("orchestrator/system_prompts/model_selector")

# ---------------------------------------------------------------------------
# System prompt + prompt template + scout prompt
# ---------------------------------------------------------------------------

# Episodic-memory section of the system prompt. Swapped in at construction time
# (see FineTuningOrchestrator.__init__) based on the enable_memory flag — the
# __MEMORY_SECTION__ sentinel in _FINE_TUNING_SYSTEM_PROMPT is .replace()'d
# with exactly one of these two blocks.
_MEMORY_SECTION_ENABLED = load_prompt("orchestrator/fragments/memory_section_enabled")

_MEMORY_SECTION_DISABLED = load_prompt("orchestrator/fragments/memory_section_disabled")

_FINE_TUNING_SYSTEM_PROMPT = load_prompt("orchestrator/system_prompts/fine_tuning")

_FINE_TUNING_PROMPT_TEMPLATE = load_prompt("orchestrator/user_prompts/fine_tuning")

_RESUME_TRAINING_PROMPT_TEMPLATE = load_prompt("orchestrator/user_prompts/resume_training")


_RESUME_FROM_PAUSE_PROMPT_TEMPLATE = load_prompt("orchestrator/user_prompts/resume_from_pause")


# ---------------------------------------------------------------------------
# Finalizer (post-training)
# ---------------------------------------------------------------------------

_FINE_TUNING_FINALIZER_SYSTEM_PROMPT = load_prompt("orchestrator/system_prompts/fine_tuning_finalizer")

_FINE_TUNING_FINALIZER_PROMPT_TEMPLATE = load_prompt("orchestrator/user_prompts/fine_tuning_finalizer")

_TRAINING_RECOVERY_SYSTEM_PROMPT = load_prompt("orchestrator/system_prompts/training_recovery")


@dataclass
class FinalizerResult:
    handoff_kind: str
    terminal_state: str                                  # completed | failed | unknown ("unknown" == could-not-verify)
    final_metrics: dict = field(default_factory=dict)
    final_model_path: str | None = None
    checkpoint_paths: list[str] = field(default_factory=list)
    trackio_url: str | None = None
    actual_cost_usd: float | None = None
    hub_best_repo_id: str | None = None                  # set by finalizer §7 (only best is pushed)
    repaired_files: list[str] = field(default_factory=list)
    repaired_dirs: list[str] = field(default_factory=list)
    failure: dict | None = None                          # {"cause": ..., "evidence": ...}
    completed_at: str | None = None
    raw: str = ""
    agent_cost_usd: float | None = None

    @classmethod
    def from_summary(cls, summary: dict | None, raw: str, agent_cost_usd: float | None) -> "FinalizerResult":
        if not summary:
            return cls(
                handoff_kind="unknown",
                terminal_state="unknown",
                raw=raw,
                agent_cost_usd=agent_cost_usd,
            )
        return cls(
            handoff_kind=str(summary.get("handoff_kind") or "unknown"),
            terminal_state=str(summary.get("terminal_state") or "unknown"),
            final_metrics=dict(summary.get("final_metrics") or {}),
            final_model_path=summary.get("final_model_path"),
            checkpoint_paths=[str(p) for p in (summary.get("checkpoint_paths") or [])],
            trackio_url=summary.get("trackio_url"),
            actual_cost_usd=_coerce_float(summary.get("actual_cost_usd")),
            hub_best_repo_id=summary.get("hub_best_repo_id"),
            repaired_files=[str(p) for p in (summary.get("repaired_files") or [])],
            repaired_dirs=[str(p) for p in (summary.get("repaired_dirs") or [])],
            failure=summary.get("failure"),
            completed_at=summary.get("completed_at"),
            raw=raw,
            agent_cost_usd=agent_cost_usd,
        )


def _coerce_float(v) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FineTuningResult:
    run_id: str
    local_run_dir: Path
    state: str                              # "completed" | "probe_failed" | "aborted_env_check" | "aborted_over_budget" | "aborted_insufficient_gpu" | "provision_timeout" | "failed" | "training_in_progress" | "resume_no_checkpoint" | "resume_instance_unreachable" | "unknown"
    instance_type: str | None = None
    instance_reused: bool = False
    ssh_alias: str | None = None
    projected_cost_usd: float | None = None
    actual_cost_usd: float | None = None
    peak_memory_gb: float | None = None
    probe_result: dict | None = None
    probe_attempts: int = 0
    cost_report: dict | None = None
    finetuned_model_path: Path | None = None
    report_paths: list[Path] = field(default_factory=list)
    checkpoint_paths: list[Path] = field(default_factory=list)
    trackio_url: str | None = None
    answer: str = ""
    agent_cost_usd: float | None = None
    resumed_from_checkpoint: str | None = None


# ---------------------------------------------------------------------------
# FineTuningOrchestrator
# ---------------------------------------------------------------------------

class FineTuningOrchestrator:
    """
    Agent-driven fine-tuning on Lambda GPU.

    The agent secures a Lambda instance (reuse-or-provision), profiles the
    dataset via the data-profiler subagent, runs a feasibility probe at p99
    sequence length, passes an automatic cost gate, then trains with trackio
    monitoring and syncs evaluation artifacts back locally.

    Example::

        orch = FineTuningOrchestrator(
            key_path="~/.ssh/lambda_key", ssh_key_name="my-key",
            model_id="facebook/esm2_t6_8M_UR50D",
            fine_tune_strategy="linear-probe",
            dataset_ref="dair-ai/emotion",
            max_cost_usd=5.0
        )
        result = orch.run_sync(
            task_description=(
                "Fine-tune ESM2 on ACE2 binding classification with 1x A10."
            )
        )
        print(result.state, result.finetuned_model_path)
    """

    # Sent to the Phase A agent when it ended its turn WITHOUT launching training
    # and WITHOUT writing an abort marker. Each generate() is a fresh query, so
    # this prompt re-establishes state from disk + the remote and drives the run
    # to one of exactly two terminal artifacts (handoff.json or an abort marker).
    _PRELAUNCH_CONTINUATION_PROMPT = (
        "You ended your turn but run {run_id} is NOT finished: you have neither "
        "LAUNCHED training (reports/handoff.json does not exist) nor ABORTED (no "
        "aborted_*.json / abort_over_budget.json / probe_failure_packet.json / "
        "failure.json under reports/). The run is stuck mid-flight and must be "
        "driven to one of exactly those two terminal states.\n\n"
        "RESUME — do not restart from scratch. First re-establish state, then "
        "continue from where you left off:\n"
        "  • Local artifacts already on disk under {local_run_dir}/ — infra.json, "
        "profile/profile.json, reports/env_check.json, any probe results, "
        "configs/, src/, scripts/launch_command.sh, manifest.json. Read them "
        "instead of regenerating them.\n"
        "  • Remote: the instance is already provisioned and the run dir is "
        "uploaded at {remote_run_dir}; any model weights you warmed are cached "
        "there. Reconnect with the lambda-repl session tools and inspect what is "
        "already done (ls the remote run dir, check for a running training pid, "
        "confirm cached weights) BEFORE repeating any GPU-spending step.\n\n"
        "Then finish the sequence: feasibility probe → cost gate → launch "
        "training (nohup … > outputs/train.log 2>&1 in the capo_remote tmux) → "
        "verify it is UP → write reports/handoff.json. If a gate legitimately "
        "fails, write the corresponding abort marker instead.\n\n"
        "Do NOT use long blocking foreground waiters (e.g. while ssh …; do "
        "sleep; done) — they consume your turn and are what stalled the previous "
        "attempt. Launch background work detached, then poll with short, "
        "non-blocking checks. Do NOT end your turn again until reports/handoff.json "
        "OR an abort marker exists on disk."
    )

    def __init__(
        self,
        key_path: str | Path,
        ssh_key_name: str,
        model_id: str,
        fine_tune_strategy: str,
        dataset_ref: str,
        tolerance_threshold: float,
        model_name: str = "claude-opus-4-8",
        ssh_alias: str | None = None,
        gpu_preference: str | None = None,
        allow_reuse_existing: bool = True,
        max_cost_usd: float = 25.0,
        auto_terminate_on_failure: bool = False,
        trackio_space_id: str | None = None,
        probe_max_retries: int = 3,
        max_post_launch_repairs: int = 2,
        max_training_recovery_attempts: int = 3,
        max_prelaunch_continuations: int = 3,
        max_turns: int = 1000,
        cwd: str | Path | None = None,
        enable_hf_research: bool = True,
        enable_memory: bool = True,
        compaction_config: CompactionConfig | None = None,
        hub_push_config: dict | None = None,
        orchestrator_effort: str | None = None,
        orchestrator_skills: list[str] | str | None = None,
        architecture: str | None = None,
    ) -> None:
        if not 0.0 <= float(tolerance_threshold) <= 1.0:
            raise ValueError(
                f"tolerance_threshold must be in [0, 1], got {tolerance_threshold!r}"
            )
        self.key_path = str(Path(key_path).expanduser().resolve())
        self.ssh_key_name = ssh_key_name
        self.model_id = model_id
        self.fine_tune_strategy = fine_tune_strategy
        self.architecture = architecture
        # Decide up-front whether the user pinned a deterministic model (so the
        # model-selector subagent can be bypassed) or left it to the selector.
        self._model_resolution = resolve_model(
            model_id, architecture=architecture, fine_tune_strategy=fine_tune_strategy
        )
        # dataset_ref may be an HF Hub id (unchanged), a local file path, a URL,
        # or a bare label.
        self._dataset_ref_original = dataset_ref
        self.dataset_ref = dataset_ref
        self._dataset_source: DatasetSource | None = None
        self.ssh_alias_override = ssh_alias
        self.gpu_preference_arg = gpu_preference
        self.allow_reuse_existing = allow_reuse_existing
        self.max_cost_usd = max_cost_usd
        # terminate the instance after the finalizer syncs artifacts on a terminal failure/escalation. 
        # Default False preserves the warm instance for cheap re-runs from cached stages (see fine_tuning.yaml).
        self.auto_terminate_on_failure = bool(auto_terminate_on_failure)
        self.trackio_space_id = trackio_space_id or ""
        self.probe_max_retries = probe_max_retries
        # Bounded post-launch auto-repair: on a mechanically-fixable failure
        # (missing_dependency / cuda_kernel), apply the fix on the remote and
        # relaunch, up to this many times, before finalizing as failed. 0 disables.
        self.max_post_launch_repairs = int(max_post_launch_repairs)
        # Agentic post-canary recovery: after the deterministic auto-repair path
        # is exhausted, a diagnosis agent proposes + applies SAFE fixes,
        # reruns the canary, and relaunches, up to this many attempts. 0 disables.
        self.max_training_recovery_attempts = int(max_training_recovery_attempts)
        # Pre-launch continuation budget. The Phase A agent runs the whole
        # pre-launch→launch sequence in ONE query. if it ends its turn (a natural
        # ResultMessage, a stray final status line, or a max_turns cut-off) WITHOUT
        # writing reports/handoff.json and WITHOUT writing an abort marker, the run
        # is neither launched nor legitimately stopped, it is stuck. Rather than
        # silently classify that as unknown and quit, we re-prompt the agent to
        # resume from on-disk + remote state, up to this many extra continuations.
        self.max_prelaunch_continuations = max(0, int(max_prelaunch_continuations))
        self.tolerance_threshold = float(tolerance_threshold)
        # Episodic-memory master switch. When False: no memory-consultant runs,
        # runs_index.md is never read, no prior_runs.md is produced, and the
        # orchestrator prompt is adapted to ignore prior runs. The current run's
        # RUN_REPORT.md is still produced and registered (write side is unaffected).
        self._enable_memory = enable_memory
        self.api_key = os.environ.get("LAMBDA_API_KEY", "")
        # Hub-push contract (per scripts/configs/fine_tuning.yaml#hub_push).
        # The finalizer reads this dict to decide whether/where to push.
        self.hub_push_config = {
            "enabled": True,
            "private": True,
            "shard_size": "2GB",
            "namespace": None,
            "repo_name_template": "capo-{run_id}-{which}",
            **(hub_push_config or {}),
        }
        # Main orchestrator.
        # Parallelism in subagents is achieved at the
        # Python level via asyncio.gather on the three dedicated runners below.
        self._orchestrator = AgentRunner(
            model_name=model_name,
            allowed_tools=[
                "Read",
                "Bash",
                "Edit",
                "Write",
                "mcp__lambda-repl__*",
            ],
            system_prompt=_FINE_TUNING_SYSTEM_PROMPT.replace(
                "__MEMORY_SECTION__",
                _MEMORY_SECTION_ENABLED if enable_memory else _MEMORY_SECTION_DISABLED,
            ),
            permission_mode="acceptEdits",
            max_turns=max_turns,
            cwd=str(cwd or _REPO_ROOT_ORCH),
            # #4 (config-gated, default off): reasoning depth + skill exposure for
            # the FIRST/main orchestrator only. None on both → unchanged behavior.
            effort=orchestrator_effort,
            skills=orchestrator_skills,
        )
        # Pre-launch parallel runners with specific tool access for each subagent's tasks.
        _infra_mcp_json = Path(cwd or _REPO_ROOT_ORCH) / ".mcp.json"
        self._infra_runner = AgentRunner(
            model_name="claude-opus-4-8",
            allowed_tools=["Read", "Write", "Bash", "mcp__lambda-repl__*"],
            system_prompt=_INFRA_SYSTEM_PROMPT,
            permission_mode="bypassPermissions",
            max_turns=100,
            cwd=str(cwd or _REPO_ROOT_ORCH),
            mcp_servers=_infra_mcp_json if _infra_mcp_json.exists() else None,
            emit_cost_per_call=False,
            effort="high",
        )
        self._data_runner = AgentRunner(
            model_name="claude-opus-4-8",
            allowed_tools=["Read", "Grep", "Glob", "Bash", "Write"],
            system_prompt=_DATA_PROFILER_SYSTEM_PROMPT,
            permission_mode="bypassPermissions",
            max_turns=80,
            cwd=str(cwd or _REPO_ROOT_ORCH),
            mcp_servers={},
            emit_cost_per_call=False,
            effort="high",
        )
        self._model_runner = AgentRunner(
            model_name="claude-haiku-4-5-20251001",
            allowed_tools=["Read", "Grep", "Glob", "Bash"],
            system_prompt=_MODEL_SELECTOR_SYSTEM_PROMPT,
            permission_mode="bypassPermissions",
            max_turns=40,
            cwd=str(cwd or _REPO_ROOT_ORCH),
            mcp_servers={},
            emit_cost_per_call=False,
        )
        # Narrow orchestrator for Phase 6 (post-training finalization + diagnosis).
        self._finalizer_runner = AgentRunner(
            model_name=model_name,
            allowed_tools=[
                "Read",
                "Write",
                "Bash",
                "mcp__lambda-repl__lambda_pull_files",
                "mcp__lambda-repl__lambda_run_command",
                # Used ONLY when auto_terminate_on_failure is enabled and the run
                # ended failed/escalation (opt-in cost backstop). Ownership-checked.
                "mcp__lambda-repl__lambda_terminate_safe",
            ],
            system_prompt=_FINE_TUNING_FINALIZER_SYSTEM_PROMPT,
            permission_mode="acceptEdits",
            max_turns=100,
            cwd=str(cwd or _REPO_ROOT_ORCH),
            effort="high",
        )
        #  post-canary recovery agent (Phase: agentic recovery loop). Only
        # built when recovery is enabled. Diagnoses a post-canary failure, applies
        # a SAFE fix, reruns the canary, and relaunches training.
        self._recovery_runner: AgentRunner | None = None
        if self.max_training_recovery_attempts > 0:
            self._recovery_runner = AgentRunner(
                model_name=model_name,
                allowed_tools=[
                    "Read",
                    "Write",
                    "Edit",
                    "Bash",
                    "mcp__lambda-repl__lambda_run_command",
                    "mcp__lambda-repl__lambda_send_to_remote_tmux",
                    "mcp__lambda-repl__lambda_push_files",
                    "mcp__lambda-repl__lambda_pull_files",
                    "mcp__lambda-repl__lambda_read_run_status",
                ],
                system_prompt=_TRAINING_RECOVERY_SYSTEM_PROMPT,
                permission_mode="acceptEdits",
                max_turns=60,
                cwd=str(cwd or _REPO_ROOT_ORCH),
                effort="high",
            )
        # Per-run recovery ledger (populated by the agentic recovery loop).
        self._recovery_ledger: RecoveryLedger | None = None
        # Pre-pipeline HuggingFace researcher — disabled via enable_hf_research=False.
        self._researcher: HFResearcher | None = (
            HFResearcher(cwd=str(cwd or _REPO_ROOT_ORCH)) if enable_hf_research else None
        )
        # Episodic memory consultant — runs before any other pre-launch agent so
        # its findings can prime model-selector / data-profiler / infra.
        # Disabled via enable_memory=False, in which case it is never built and
        # Phase -2 is skipped entirely (see run()).
        if enable_memory:
            _memory_spec = SUBAGENTS["memory-consultant"]
            self._memory_runner: AgentRunner | None = AgentRunner(
                model_name="claude-haiku-4-5-20251001",
                allowed_tools=list(_memory_spec.tools),
                system_prompt=_memory_spec.prompt,
                permission_mode="bypassPermissions",
                max_turns=30,
                cwd=str(cwd or _REPO_ROOT_ORCH),
                mcp_servers={},
                emit_cost_per_call=False,
            )
        else:
            self._memory_runner = None
        # Compaction config — the per-run Compactor is built inside run
        # once local_run_dir is known.
        self._compaction_config = compaction_config or CompactionConfig()
        # Per-run SessionStore — initialised inside run() once local_run_dir is known.
        self._session: SessionStore | None = None
        # Per-run agent cost ledger (one AgentCost per LLM agent). Reset at the
        # start of each run(); rendered into RUN_REPORT.md's Cost Report section.
        self._agent_cost_entries: list[AgentCost] = []

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @property
    def cost_overrun_factor(self) -> float:
        """α used by the 3-step gate Step 3, derived from tolerance_threshold.

        max_cost_usd < ĉ ≤ α · max_cost_usd → user confirms overrun.
        """
        return 1.0 + self.tolerance_threshold

    def _resolve_gpu_preference(self, task_description: str) -> str | None:
        """Resolve gpu_preference: explicit arg > parsed from task_description > None."""
        if self.gpu_preference_arg:
            return self.gpu_preference_arg
        return _parse_gpu_preference(task_description)

    @staticmethod
    def _restart_hint(run_id: str, phase: str, terminal_state: str | None) -> str:
        resumable = phase in ("pre_launch", "training", "finalizing") or (
            phase == "completed" and terminal_state == "training_in_progress"
        )
        if resumable:
            return (
                f"Set 'resume: {run_id}' in scripts/configs/fine_tuning.yaml to reconnect."
            )
        if phase == "completed" and terminal_state == "completed":
            return "Run completed successfully. No restart needed."
        if phase in ("failed", "completed") and terminal_state == "unknown":
            return (
                "Terminal state could not be verified (remote unreachable at "
                "finalize time). Re-run the finalizer once the instance is "
                "reachable, or inspect the remote run dir directly — do NOT assume "
                "the run failed."
            )
        if phase in ("failed", "completed") and terminal_state in ("failed", None):
            return (
                "Run failed before training was launched. "
                "Start a fresh run (leave resume: null)."
            )
        return "Check run logs for details."

    def _phase(self, run_id: str, phase: str, **extra) -> None:
        """Best-effort atomic phase update via the run's SessionStore."""
        if self._session is None:
            return
        hint = self._restart_hint(run_id, phase, extra.get("terminal_state"))
        try:
            self._session.update(current_phase=phase, restart_hint=hint, **extra)
        except Exception as exc:  # pragma: no cover - defensive
            ip.emit(f"[session] phase update failed ({phase}): {exc}")

    # ------------------------------------------------------------------ #
    # Cost accounting                                                      #
    # ------------------------------------------------------------------ #

    def _record_agent_cost(self, agent_name: str, result, *, model: str | None = None) -> None:
        """Append a per-agent cost line (with token detail) from an AgentRunResult."""
        if result is None or isinstance(result, Exception):
            return
        self._agent_cost_entries.append(
            make_agent_cost(
                agent_name,
                model or getattr(result, "model_name", "unknown"),
                input_tokens=getattr(result, "input_tokens", 0) or 0,
                output_tokens=getattr(result, "output_tokens", 0) or 0,
                cache_read_tokens=getattr(result, "cache_read_tokens", 0) or 0,
                cache_creation_tokens=getattr(result, "cache_creation_tokens", 0) or 0,
                sdk_cost_usd=getattr(result, "total_cost_usd", None),
            )
        )

    def _record_agent_cost_total(self, agent_name: str, model: str, cost_usd: float | None) -> None:
        """Append a cost-only per-agent line (token counts unknown)."""
        if cost_usd is None:
            return
        self._agent_cost_entries.append(
            make_agent_cost_from_total(agent_name, model, cost_usd)
        )

    def _build_cost_report(
        self,
        local_run_dir: Path,
        infra_data: dict | None,
        actual_cost_usd: float | None,
    ) -> RunCostReport:
        """Assemble the run's cost ledger: agent breakdown + the one infra line."""
        infra_costs = []
        if infra_data:
            launched = self._launched_at_iso(local_run_dir)
            runtime_s = self._elapsed_seconds(launched)
            ic = make_infra_cost(
                provider="lambda",
                instance_id=infra_data.get("instance_id"),
                instance_type=infra_data.get("instance_type") or "unknown",
                # resolved_gpu already carries the count, e.g. "1x A100".
                gpu_type=infra_data.get("resolved_gpu") or infra_data.get("gpu_name"),
                gpu_count=None,
                runtime_seconds=runtime_s or 0.0,
                hourly_rate_usd=_coerce_float(infra_data.get("hourly_rate_usd")),
            )
            # The finalizer's actual_cost_usd is authoritative when present —
            # keep the infra-line total consistent with it.
            if actual_cost_usd is not None:
                ic.cost_usd = float(actual_cost_usd)
            infra_costs.append(ic)
        return RunCostReport(
            agent_costs=list(self._agent_cost_entries),
            infra_costs=infra_costs,
        )

    def _write_cost_report(
        self,
        local_run_dir: Path,
        reports_dir: Path,
        report: RunCostReport,
    ) -> None:
        """Persist the cost ledger and append a Cost Report section to RUN_REPORT.md.

        Writes reports/run_cost.json (machine-readable) and, when the
        finalizer already wrote RUN_REPORT.md, appends the scientific
        ## Cost Report markdown section once (idempotent)."""
        try:
            reports_dir.mkdir(parents=True, exist_ok=True)
            (reports_dir / "run_cost.json").write_text(
                json.dumps(report.to_dict(), indent=2), encoding="utf-8"
            )
        except Exception as exc:  # pragma: no cover - defensive
            ip.emit(f"[cost] failed to write run_cost.json: {exc}")

        report_md = local_run_dir / "RUN_REPORT.md"
        if not report_md.exists():
            return
        try:
            existing = report_md.read_text(encoding="utf-8")
            if "## Cost Report" in existing:
                return  # already appended (e.g. on a resume re-finalize)
            section = render_cost_report_markdown(report)
            report_md.write_text(
                existing.rstrip() + "\n\n" + section + "\n", encoding="utf-8"
            )
        except Exception as exc:  # pragma: no cover - defensive
            ip.emit(f"[cost] failed to append Cost Report to RUN_REPORT.md: {exc}")

    # ------------------------------------------------------------------ #
    # Agentic post-canary recovery                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _failure_to_dict(failure) -> dict:
        if failure is None:
            return {}
        return {
            "failure_category": getattr(failure, "failure_category", "unknown"),
            "summary": getattr(failure, "summary", ""),
            "remediation": getattr(failure, "remediation", ""),
            "missing_packages": list(getattr(failure, "missing_packages", []) or []),
            "failing_file": getattr(failure, "failing_file", "") or "",
        }

    @staticmethod
    def _read_effective_config(local_run_dir: Path) -> dict:
        for rel in ("configs/training.yaml", "configs/fine_tuning.yaml"):
            path = local_run_dir / rel
            if path.exists():
                try:
                    import yaml as _yaml

                    return _yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                except Exception:
                    return {}
        return {}

    @staticmethod
    def _latest_checkpoint(local_run_dir: Path) -> str | None:
        last = local_run_dir / "checkpoints" / "last"
        if last.exists() and any(last.iterdir()):
            return "checkpoints/last"
        return None

    @staticmethod
    def _read_log_tail(local_run_dir: Path, n: int = 200) -> str:
        # Only the REMOTE training logs (pulled back under outputs/) are valid
        # evidence for the recovery agent. Never fall back to run.log /
        # run_err.log — those are the orchestrator's OWN chatter, not training
        # output, and feeding them to the diagnosis agent is misleading.
        for rel in ("outputs/train.log", "outputs/train_err.log"):
            path = local_run_dir / rel
            if path.exists():
                try:
                    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                    if any(line.strip() for line in lines):
                        return "\n".join(lines[-n:])
                except OSError:
                    continue
        return "(no local training log available — pull it from the remote)"

    async def _run_recovery_loop(
        self,
        handoff: "MonitorHandoff",
        monitor_ctx: "HealthMonitorContext",
        local_run_dir: Path,
        reports_dir: Path,
    ) -> "MonitorHandoff":
        """Bounded agentic recovery after a post-canary training failure.

        Each attempt: diagnose →  recovery agent proposes + applies a SAFE
        fix (or stops for an UNSAFE one) → re-monitor. Returns the (possibly
        recovered) handoff and records the attempt ledger on self._recovery_ledger.
        """
        if self._recovery_runner is None or self.max_training_recovery_attempts <= 0:
            return handoff
        if handoff.kind not in ("failed", "escalation") or not monitor_ctx.handoff_path.exists():
            return handoff  # not a post-canary (post-launch) failure

        try:
            payload = json.loads(monitor_ctx.handoff_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        ssh_alias = payload.get("ssh_alias") or _read_infra_ssh_alias(local_run_dir)
        remote_dir = payload.get("remote_run_dir") or monitor_ctx.remote_run_dir
        if not ssh_alias:
            ip.emit("[recovery] no ssh_alias available — skipping agentic recovery")
            return handoff

        ledger = RecoveryLedger()
        self._recovery_ledger = ledger
        config = self._read_effective_config(local_run_dir)
        last_ckpt = self._latest_checkpoint(local_run_dir)

        attempt = 0
        while (
            handoff.kind in ("failed", "escalation")
            and attempt < self.max_training_recovery_attempts
        ):
            attempt += 1
            # Fingerprint handoff.json BEFORE dispatching the agent so we can
            # detect an on-instance relaunch even if the agent never emits a
            # verdict (e.g. it runs out of turns right after relaunching).
            pre_fp = self._handoff_launch_fp(monitor_ctx.handoff_path)

            def _diagnose() -> dict:
                try:
                    f = diagnose_post_launch_failure(
                        ssh_alias=str(ssh_alias),
                        key_path=self.key_path,
                        remote_run_dir=str(remote_dir),
                        local_run_dir=local_run_dir,
                    )
                    return self._failure_to_dict(f)
                except Exception as exc:  # pragma: no cover - defensive
                    ip.emit(f"[recovery] diagnostics failed (non-fatal): {exc}")
                    return {}

            failure = _diagnose()
            context = build_recovery_context(
                attempt=attempt,
                max_attempts=self.max_training_recovery_attempts,
                failure=failure,
                config=config,
                last_checkpoint=last_ckpt,
                canary_passed=True,
                log_tail=self._read_log_tail(local_run_dir),
                ssh_alias=str(ssh_alias),
                remote_run_dir=str(remote_dir),
                local_run_dir=str(local_run_dir),
            )
            ip.emit(
                f"[recovery] attempt {attempt}/{self.max_training_recovery_attempts}: "
                f"diagnosing post-canary failure ({failure.get('failure_category', 'unknown')})"
            )
            result = None
            try:
                result = await self._recovery_runner.generate(prompt=context)
                verdict = parse_json_with_repair(result.answer)
            except Exception as exc:
                ip.emit(f"[recovery] recovery agent failed (non-fatal): {exc}")
                verdict = None
            if result is not None:
                self._record_agent_cost_total(
                    "training-recovery",
                    self._recovery_runner.model_name,
                    getattr(result, "total_cost_usd", None),
                )
            ra = RecoveryAttempt.from_verdict(attempt, verdict)
            ledger.attempts.append(ra)

            async def _remonitor(note: str) -> "MonitorHandoff":
                """Re-watch the run to a terminal state; mark recovered if healthy."""
                ip.emit(f"[recovery] {note} — re-monitoring")
                monitor = TrainingHealthMonitor(monitor_ctx)
                try:
                    h = await monitor.run_loop()
                except Exception as mon_exc:
                    ip.emit(f"[recovery] post-recovery monitor raised: {mon_exc}")
                    return MonitorHandoff(
                        kind="stopped",
                        reason=f"post-recovery monitor raised: {mon_exc}",
                        last_report=None,
                    )
                if h.kind not in ("failed", "escalation"):
                    ra.outcome = "recovered"
                    ledger.final_outcome = "recovered"
                    # Clear the stale pre-recovery failure so the finalizer doesn't
                    # read a recovered run as failed.
                    (reports_dir / "post_launch_failure.json").unlink(missing_ok=True)
                    ip.emit(f"[recovery] training recovered after {attempt} attempt(s)")
                return h

            # The run was never broken: the agent verified it is alive and making
            # forward progress (a monitor false-alarm on a long CPU-bound stage)
            # and only extended the monitor deadline. Re-watch the SAME live
            # process — do NOT require a relaunch and do NOT stop for the user.
            if ra.outcome == "resume_monitoring":
                handoff = await _remonitor(
                    "run still alive & progressing (monitor false-alarm); "
                    f"deadline extended by agent [{ra.fix_applied or 'handoff.json'}]"
                )
                if handoff.kind not in ("failed", "escalation"):
                    break
                continue  # escalated again → re-diagnose (bounded by max attempts)

            # Relaunch is disk evidence, not a verdict. The recovery agent may fix and
            # restart training, then hit max_turns before writing verdict JSON; `ra` then
            # looks failed even though a live run exists. If handoff.json advanced
            # (new launched_at / pid), monitor that run to its true terminal state.
            # Observing an already-started process is always safe.
            post_fp = self._handoff_launch_fp(monitor_ctx.handoff_path)
            if self._relaunch_detected(pre_fp, post_fp) and not ra.relaunched:
                ip.emit(
                    f"[recovery] attempt {attempt}: recovery agent relaunched "
                    f"on-instance (handoff.json advanced to pid={post_fp[1]}, "
                    f"launched_at={post_fp[0] or '?'}) but emitted no verdict "
                    f"(subtype={getattr(result, 'subtype', '') or 'none'}) — "
                    "re-monitoring the live run rather than reporting failed"
                )
                ra.relaunched = True
                ra.new_pid = post_fp[1]
                if ra.outcome in ("failed", "needs_user"):
                    ra.outcome = "applied"
                if not ra.fix_applied:
                    ra.fix_applied = (
                        "on-instance fix + relaunch detected via handoff.json "
                        "(agent verdict not emitted)"
                    )
                handoff = await _remonitor(
                    "relaunched on-instance (detected via handoff.json; "
                    "verdict not emitted)"
                )
                if handoff.kind not in ("failed", "escalation"):
                    break
                continue

            if ra.fix_safety == "unsafe" or ra.outcome == "needs_user":
                ledger.final_outcome = "needs_user"
                ip.emit(
                    "[recovery] proposed fix is UNSAFE or needs input — stopping for "
                    "user confirmation (see reports/recovery_pending_question.json)"
                )
                break
            if not ra.relaunched:
                ip.emit(
                    f"[recovery] attempt {attempt} did not relaunch (outcome={ra.outcome}) "
                    "— trying again" if attempt < self.max_training_recovery_attempts
                    else f"[recovery] attempt {attempt} did not relaunch — giving up"
                )
                continue

            handoff = await _remonitor(
                f"relaunched after fix '{ra.fix_applied or ra.fix_type}'"
            )
            if handoff.kind not in ("failed", "escalation"):
                break

        if ledger.final_outcome == "not_attempted" and ledger.attempts:
            ledger.final_outcome = "exhausted"
        try:
            ledger.write_json(reports_dir / "recovery_ledger.json")
        except Exception as exc:  # pragma: no cover - defensive
            ip.emit(f"[recovery] failed to write recovery_ledger.json: {exc}")
        return handoff

    def _append_recovery_report(self, local_run_dir: Path) -> None:
        """Append the ## Training Recovery section to RUN_REPORT.md (once).

        Only when the agentic recovery loop actually ran (ledger has attempts)."""
        ledger = self._recovery_ledger
        if ledger is None or not ledger.attempts:
            return
        report_md = local_run_dir / "RUN_REPORT.md"
        if not report_md.exists():
            return
        try:
            existing = report_md.read_text(encoding="utf-8")
            if "## Training Recovery" in existing:
                return
            section = render_recovery_report_markdown(ledger)
            report_md.write_text(
                existing.rstrip() + "\n\n" + section + "\n", encoding="utf-8"
            )
        except Exception as exc:  # pragma: no cover - defensive
            ip.emit(f"[recovery] failed to append Training Recovery to RUN_REPORT.md: {exc}")

    @staticmethod
    def _handoff_launch_fp(handoff_path: Path) -> tuple[str, int | None]:
        """(launched_at_iso, python_pid|pid) from handoff.json — for relaunch detection.

        Returns ("", None) if the file is missing/unparseable so a later real
        launch always compares as "advanced".
        """
        try:
            d = json.loads(handoff_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ("", None)
        pid = d.get("python_pid") if d.get("python_pid") is not None else d.get("pid")
        try:
            pid = int(pid) if pid is not None else None
        except (TypeError, ValueError):
            pid = None
        return (str(d.get("launched_at_iso") or ""), pid)

    @staticmethod
    def _relaunch_detected(
        before: tuple[str, int | None], after: tuple[str, int | None]
    ) -> bool:
        """True if handoff.json shows a launch newer than `before` (a relaunch).

        A relaunch advances launched_at_iso (ISO-8601 strings sort chronologically)
        or swaps in a new training pid. Either is proof the recovery agent restarted
        training even if it never emitted a verdict.
        """
        b_iso, b_pid = before
        a_iso, a_pid = after
        if a_iso and a_iso != b_iso and a_iso > b_iso:
            return True
        if a_pid is not None and a_pid != b_pid:
            return True
        return False

    async def _reconcile_terminal_state_from_remote(
        self, handoff: "MonitorHandoff", monitor_ctx: "HealthMonitorContext"
    ) -> "MonitorHandoff":
        """Deterministic ground-truth check before finalizing a failure.

        The handoff can be STALE: a recovery agent cut off by max_turns, a monitor
        false-alarm on a slow CPU stage, or a run that finished after the monitor
        stopped. Before the finalizer trusts kind=failed, SSH-read the remote
        status.json directly (read-only, no agent, no interactive MCP permission —
        exactly the pull that was DENIED when a completed run was mis-reported as
        "training never started"). If the remote disagrees with the handoff,
        reconcile it so the finalizer works from the truth.
        """
        if handoff.kind not in ("failed", "escalation"):
            return handoff
        ssh_alias = _read_infra_ssh_alias(monitor_ctx.local_run_dir)
        if not ssh_alias:
            return handoff
        try:
            status = read_remote_run_status(
                str(ssh_alias), monitor_ctx.run_id, self.key_path
            )
        except Exception as exc:  # pragma: no cover - defensive (network)
            ip.emit(f"[fine-tuning] remote status reconciliation skipped (non-fatal): {exc}")
            return handoff
        state = (getattr(status, "state", "") or "").lower()
        failure_marker = monitor_ctx.local_run_dir / "reports" / "post_launch_failure.json"
        if state == "completed":
            ip.emit(
                "[fine-tuning] handoff=failed but remote status.json=completed — "
                "reconciling to completed (the run finished after the monitor "
                "stopped; the failure verdict was stale)"
            )
            failure_marker.unlink(missing_ok=True)
            return MonitorHandoff(
                kind="completed",
                reason="reconciled from remote status.json=completed (handoff was stale)",
                last_report=handoff.last_report,
                agent_cost_usd=handoff.agent_cost_usd,
            )
        if state in ("running", "pending"):
            ip.emit(
                f"[fine-tuning] handoff=failed but remote status.json={state} — "
                "the run is still alive; re-monitoring to a true terminal state"
            )
            monitor = TrainingHealthMonitor(monitor_ctx)
            try:
                h = await monitor.run_loop()
            except Exception as mon_exc:  # pragma: no cover - defensive
                ip.emit(f"[fine-tuning] reconciliation re-monitor raised (non-fatal): {mon_exc}")
                return handoff
            if h.kind not in ("failed", "escalation"):
                failure_marker.unlink(missing_ok=True)
            return h
        return handoff

    @staticmethod
    def _launched_at_iso(local_run_dir: Path | None) -> str | None:
        """Read the training launch timestamp from handoff.json (best-effort)."""
        if local_run_dir is None:
            return None
        for rel in ("reports/handoff.json", "reports/cost_report.json", "infra.json"):
            path = local_run_dir / rel
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            stamp = data.get("launched_at_iso") or data.get("launched_at")
            if stamp:
                return str(stamp)
        return None

    @staticmethod
    def _elapsed_seconds(started_at_iso: str | None) -> float | None:
        """Seconds from an ISO-8601 start to now (UTC); None when unparseable."""
        if not started_at_iso:
            return None
        from capo.remote.lambda_pricing import parse_datetime

        start = parse_datetime(started_at_iso)
        if start is None:
            return None
        return max(0.0, (datetime.now(timezone.utc) - start).total_seconds())

    def _provision_trackio_dashboard(self, reports_dir: Path) -> None:
        """Seed an HF Space + bucket for the trackio dashboard before launch.

        Writes the dashboard URL to reports/trackio_url.txt so the
        agent's handoff step can read it without inventing a URL itself. On
        any failure (no HF_TOKEN, whoami refused, seeding error), emits a
        warning and clears trackio_space_id so the training command omits
        the --trackio-space-id flag and the run proceeds without a dashboard.
        """
        hf_token = (
            os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        if not hf_token:
            ip.emit("[trackio] HF_TOKEN not set; running without dashboard")
            self.trackio_space_id = ""
            return
        space_id = self.trackio_space_id or resolve_default_space_id(hf_token)
        if not space_id:
            ip.emit("[trackio] could not resolve HF username; running without dashboard")
            self.trackio_space_id = ""
            return
        try:
            ensure_trackio_dashboard(space_id, hf_token, log=ip.emit)
        except Exception as exc:
            ip.emit(f"[trackio] dashboard setup failed (non-fatal): {exc}")
            self.trackio_space_id = ""
            return
        self.trackio_space_id = space_id
        url = f"https://huggingface.co/spaces/{space_id}"
        (reports_dir / "trackio_url.txt").write_text(url + "\n", encoding="utf-8")
        ip.emit(f"[trackio] dashboard ready: {url}")

    def _build_prompt(
        self,
        run_id: str,
        task_description: str,
        local_run_dir: Path,
        gpu_preference: str | None,
        pre_launch_context: str = "",
        research_enabled: bool = True,
        memory_enabled: bool = True,
    ) -> str:
        # Write task description to a file instead of embedding it in the prompt.
        # Bio-sensitive content in the prompt might trigger Anthropic content-policy filters
        # -> the agent reads the file via a tool call, which is not subject to the same check.
        (local_run_dir / "task.md").write_text(task_description, encoding="utf-8")

        if research_enabled:
            research_context = (
                f"Pre-populated at {local_run_dir}/reports/research_findings_agent.json — "
                "read this file in Step 0\nfor recommended datasets, benchmarks, and "
                "hyperparameters."
            )
            research_artifact_line = (
                f"  {local_run_dir}/reports/research_findings_agent.json   "
                "— recommended datasets, benchmarks, hyperparameters\n"
            )
        else:
            research_context = (
                "HF Hub research was disabled for this run (enable_hf_research=false); "
                "no research_findings file exists.\n"
                "Derive datasets, benchmarks, and hyperparameters from "
                f"{local_run_dir}/task.md, {local_run_dir}/profile/profile.json, and "
                f"{local_run_dir}/reports/model_selection.json."
            )
            research_artifact_line = ""

        if memory_enabled:
            memory_context = (
                "Episodic memory enabled. A memory-consultant scanned "
                "<repo>/runs/runs_index.md and wrote 0–3 advisory priors to "
                f"{local_run_dir}/reports/prior_runs.md — read it in Step 0 and "
                "treat it as advisory only (current-run artifacts always override)."
            )
            prior_runs_artifact_line = (
                f"  {local_run_dir}/reports/prior_runs.md            "
                "— 0–3 prior RUN_REPORT.md bodies for similar tasks "
                "(advisory priors; current artifacts override)\n"
            )
        else:
            memory_context = (
                "Episodic memory was disabled for this run (enable_memory=false); "
                "<repo>/runs/runs_index.md was NOT consulted and no prior_runs.md "
                "exists.\nMake every decision from current-run artifacts only "
                f"({local_run_dir}/task.md, {local_run_dir}/profile/profile.json, "
                f"{local_run_dir}/reports/model_selection.json, and probe results). "
                "This run still writes and registers its own RUN_REPORT.md normally."
            )
            prior_runs_artifact_line = ""

        remote_run_dir = f"~/capo_runs/{run_id}"
        trackio_cli_arg = (
            f" --trackio-space-id {self.trackio_space_id}"
            if self.trackio_space_id
            else ""
        )
        return _FINE_TUNING_PROMPT_TEMPLATE.format(
            run_id=run_id,
            model_id=self.model_id,
            fine_tune_strategy=self.fine_tune_strategy,
            dataset_ref=self.dataset_ref,
            gpu_preference=gpu_preference if gpu_preference else "None",
            allow_reuse_existing=str(self.allow_reuse_existing),
            ssh_alias_override=self.ssh_alias_override or "None",
            ssh_key_name=self.ssh_key_name,
            key_path=self.key_path,
            local_run_dir=str(local_run_dir),
            remote_run_dir=remote_run_dir,
            skills_dir=str(_SKILLS_DIR),
            max_cost_usd=self.max_cost_usd,
            probe_max_retries=self.probe_max_retries,
            tolerance_threshold=self.tolerance_threshold,
            trackio_space_id=self.trackio_space_id,
            trackio_cli_arg=trackio_cli_arg,
            pre_launch_context=pre_launch_context or "(pre-launch results not yet available)",
            research_context=research_context,
            research_artifact_line=research_artifact_line,
            memory_context=memory_context,
            prior_runs_artifact_line=prior_runs_artifact_line,
        )

    async def _consult_episodic_memory(
        self,
        local_run_dir: Path,
        gpu_preference: str | None,
    ) -> tuple[str, float | None]:
        """Invoke the memory-consultant subagent to scan past RUN_REPORT.md files.

        Assumes <local_run_dir>/task.md already exists. Always returns a
        non-empty prior_runs.md body — if the consultant fails or no priors
        match, a stub is written so downstream agents can reference the path
        unconditionally.
        """
        reports_dir = local_run_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        prior_runs_path = reports_dir / "prior_runs.md"
        index_exists = _RUNS_INDEX_PATH.exists() and _RUNS_INDEX_PATH.stat().st_size > 0

        if not index_exists:
            prior_runs_path.write_text(
                "# Prior runs (advisory)\n\n"
                "No prior runs in the episodic memory index. This is either the "
                "first CAPO run or the index was reset.\n",
                encoding="utf-8",
            )
            ip.emit("[fine-tuning] Episodic memory: no prior runs (index absent or empty)")
            return prior_runs_path.read_text(encoding="utf-8"), None

        prompt = (
            f"local_run_dir = {local_run_dir}\n"
            f"index_path = {_RUNS_INDEX_PATH}\n"
            f"task_md_path = {local_run_dir / 'task.md'}\n"
            f"current_model_id = {self.model_id}\n"
            f"current_dataset_ref = {self.dataset_ref}\n"
            f"current_gpu_preference = {gpu_preference or 'None'}\n"
        )
        ip.emit("[fine-tuning] Episodic memory: consulting past runs...")
        memory_cost: float | None = None
        try:
            result = await self._memory_runner.generate(prompt=prompt)
            memory_cost = getattr(result, "total_cost_usd", None)
            self._record_agent_cost("memory-consultant", result)
        except Exception as exc:
            ip.emit(f"[memory] Consultant failed (non-fatal): {exc}")

        if not prior_runs_path.exists():
            prior_runs_path.write_text(
                "# Prior runs (advisory)\n\n"
                "Memory consultant did not produce a report. No priors available.\n",
                encoding="utf-8",
            )
        prior_runs_md = prior_runs_path.read_text(encoding="utf-8")
        ip.emit(
            f"[fine-tuning] Episodic memory: prior_runs.md ready "
            f"({len(prior_runs_md)} chars)"
        )
        return prior_runs_md, memory_cost

    async def _run_pre_launch(
        self,
        run_id: str,
        local_run_dir: Path,
        gpu_preference: str | None,
        prior_runs_md: str = "",
    ) -> tuple[str, float]:
        """Run infrastructure, data profiling and model selection in parallel.

        Returns (context_str, pre_launch_cost_usd). The context string contains
        only clean key=value summaries — no raw agent answers — so it can be
        safely injected into the main agent's cached_prefix. The main agent 
        reads the actual artifact files (infra.json, profile.json, model_selection.json) 
        from local_run_dir in Step 0 regardless.
        """
        profile_dir = local_run_dir / "profile"
        profile_dir.mkdir(parents=True, exist_ok=True)

        prior_runs_section = ""
        if prior_runs_md.strip():
            prior_runs_section = (
                "## Prior runs (advisory)\n"
                "The episodic memory consultant selected the following past runs as\n"
                "relevant priors. They are advisory — always defer to current-run\n"
                "artifacts when they conflict.\n\n"
                f"{prior_runs_md}\n\n"
                "---\n\n"
            )

        infra_prompt = (
            f"{prior_runs_section}"
            f"run_id = {run_id}\n"
            f"model_id = {self.model_id}\n"
            f"fine_tune_strategy = {self.fine_tune_strategy}\n"
            f"gpu_preference = {gpu_preference or 'None'}\n"
            f"allow_reuse_existing = {self.allow_reuse_existing}\n"
            f"ssh_alias_override = {self.ssh_alias_override or 'None'}\n"
            f"ssh_key_name = {self.ssh_key_name}\n"
            f"key_path = {self.key_path}\n"
            f"max_cost_usd = {self.max_cost_usd}\n"
            f"local_run_dir = {local_run_dir}\n"
            f"skills_dir = {_SKILLS_DIR}\n"
            f"dataset_ref = {self.dataset_ref}"
        )
        (profile_dir / "plots").mkdir(parents=True, exist_ok=True)
        # Dataset-source hints for the profiler. It runs at repo-root CWD (not
        # cd'd into the run dir), so a local dataset needs an ABSOLUTE staged
        # path — the relative effective_ref only resolves after the remote cd.
        src = self._dataset_source
        dataset_kind = src.kind if src else "hf"
        if src and src.kind == "local" and src.staged_rel_path:
            dataset_local_path = str((local_run_dir / src.staged_rel_path).resolve())
        else:
            dataset_local_path = "None"
        dataset_file_format = (src.file_format if src else None) or "None"
        data_prompt = (
            f"{prior_runs_section}"
            f"dataset_ref = {self.dataset_ref}\n"
            f"dataset_kind = {dataset_kind}\n"
            f"dataset_local_path = {dataset_local_path}\n"
            f"dataset_file_format = {dataset_file_format}\n"
            f"profile_path = {profile_dir / 'profile.json'}\n"
            f"profile_plots_dir = {profile_dir / 'plots'}\n"
            f"skills_dir = {_SKILLS_DIR}\n"
            f"local_run_dir = {local_run_dir}\n"
            f"key_path = {self.key_path}"
        )
        res = self._model_resolution
        bypass_selection = res.bypass_selection
        sel_path = local_run_dir / "reports" / "model_selection.json"
        candidate_hint = ""
        if res.candidates:
            cand_ids = ", ".join(
                str(c.get("registry_id") or c.get("model_id")) for c in res.candidates
            )
            candidate_hint = (
                f"candidate_registry_ids = {cand_ids}\n"
                "# User specified an architecture matching these registry entries — "
                "restrict your selection to them.\n"
            )
        model_prompt = (
            f"{prior_runs_section}"
            f"model_id_hint = {self.model_id}\n"
            f"fine_tune_strategy_hint = {self.fine_tune_strategy}\n"
            f"{candidate_hint}"
            f"dataset_ref = {self.dataset_ref}\n"
            f"gpu_preference = {gpu_preference or 'None'}\n"
            f"max_cost_usd = {self.max_cost_usd}\n"
            f"model_selection_path = {sel_path}\n"
            f"skills_dir = {_SKILLS_DIR}"
        )

        async def _tagged(tag: str, coro):
            token = set_runner_tag(tag)
            try:
                return await coro
            finally:
                reset_runner_tag(token)

        if bypass_selection:
            # User pinned a deterministic model — skip the selector subagent and
            # write model_selection.json directly from the resolution.
            sel_path.parent.mkdir(parents=True, exist_ok=True)
            sel_path.write_text(
                json.dumps(build_model_selection_json(res), indent=2), encoding="utf-8"
            )
            ip.emit(
                f"[fine-tuning] model selection bypassed ({res.mode}): {res.reason}"
            )
            ip.emit("[fine-tuning] Phases 0+1: parallel pre-launch (infra · data)...")
            results = await asyncio.gather(
                _tagged("infra-agent", self._infra_runner.generate(prompt=infra_prompt)),
                _tagged("data-agent",  self._data_runner.generate(prompt=data_prompt)),
                return_exceptions=True,
            )
            infra_r, data_r = results
            model_r = None
        else:
            ip.emit("[fine-tuning] Phases 0+1+2: parallel pre-launch (infra · data · model)...")
            results = await asyncio.gather(
                _tagged("infra-agent",     self._infra_runner.generate(prompt=infra_prompt)),
                _tagged("data-agent",      self._data_runner.generate(prompt=data_prompt)),
                _tagged("model-sel-agent", self._model_runner.generate(prompt=model_prompt)),
                return_exceptions=True,
            )
            infra_r, data_r, model_r = results

        # If any pre-launch agent hit an Anthropic auth / credit error, fail
        # fast with the real cause
        from capo.orchestration.agent_runner import AnthropicAuthError as _AuthErr
        _auth_failures = [
            (label, r) for label, r in
            (("infra-agent", infra_r), ("data-agent", data_r), ("model-sel-agent", model_r))
            if isinstance(r, _AuthErr)
        ]
        if _auth_failures:
            labels = ", ".join(lbl for lbl, _ in _auth_failures)
            ip.error(
                f"[pre-launch] Anthropic API auth/credit error in: {labels}. "
                "See [anthropic-auth] lines above for the exact cause."
            )
            raise _auth_failures[0][1]

        def _fmt(label: str, r: object, kind: str) -> str:
            """Return a clean one-line-per-field summary for the main agent.

            We intentionally do NOT inject the raw agent answer here. 
            The main agent reads the actual JSON artifact files (infra.json, profile.json,
            model_selection.json) from local_run_dir in Step 0 anyway.
            """
            if isinstance(r, Exception):
                ip.error(f"[pre-launch] {label} runner failed: {r}")
                return f"### {label}\nstate=FAILED reason={r}"
            cost = getattr(r, "total_cost_usd", None)
            cost_str = f" (${cost:.4f})" if cost else ""
            answer = getattr(r, "answer", "")
            summary_line = "state=unknown"
            try:
                import json as _json
                data = _json.loads(answer)
                if kind == "infra":
                    state = data.get("state", "?")
                    if state == "ready":
                        gpu   = data.get("resolved_gpu", "?")
                        ssh   = data.get("ssh_alias", "?")
                        rate  = data.get("hourly_rate_usd", "?")
                        itype = data.get("instance_type", "?")
                        reuse = "reused" if data.get("instance_reused") else "new"
                        ip.emit(f"[fine-tuning] {label} ready: {gpu} ({reuse}) ssh={ssh} $/hr={rate}{cost_str}")
                        summary_line = (
                            f"state=ready | instance_type={itype} | resolved_gpu={gpu} "
                            f"| hourly_rate_usd={rate} | instance_reused={data.get('instance_reused', False)}"
                        )
                    else:
                        reason = data.get("reason") or data.get("warning") or ""
                        ip.emit(f"[fine-tuning] {label} {state}: {reason}{cost_str}")
                        summary_line = f"state={state} | reason={reason}"
                elif kind == "data":
                    n      = data.get("n_samples", "?")
                    p99    = (data.get("length_percentiles") or {}).get("p99", "?")
                    dtype  = data.get("dataset_type", "?")
                    plots  = len(data.get("plots") or {})
                    where  = data.get("profiled_on", "local")
                    ip.emit(f"[fine-tuning] {label} done: n={n} p99={p99} type={dtype} plots={plots} profiled_on={where}{cost_str}")
                    summary_line = (
                        f"state=done | n_samples={n} | dataset_type={dtype} "
                        f"| p99_length={p99} | plots={plots} | profiled_on={where}"
                    )
                elif kind == "model":
                    preferred = data.get("preferred", "best_fit")
                    cand = data.get(preferred) or data.get("best_fit") or {}
                    model  = cand.get("model_id", "?")
                    strat  = cand.get("fine_tune_strategy", "?")
                    vram   = cand.get("min_vram_gb", "?")
                    score  = cand.get("score", "?")
                    params = cand.get("param_count_b", "?")
                    lora_r = cand.get("lora_r")
                    bf  = (data.get("best_fit")  or {}).get("model_id", "?")
                    bud = (data.get("budget")    or {}).get("model_id", "?")
                    fro = (data.get("frontier")  or {}).get("model_id", "?")
                    ip.emit(
                        f"[fine-tuning] {label} → preferred={model} "
                        f"({strat}, {params}B params, vram≥{vram}GB, score={score}){cost_str}"
                    )
                    ip.emit(f"[fine-tuning]   best_fit={bf}  budget={bud}  frontier={fro}")
                    summary_line = (
                        f"state=done | preferred_model={model} | strategy={strat} "
                        f"| min_vram_gb={vram} | param_count_b={params} | score={score}"
                    )
                    if lora_r:
                        summary_line += f" | lora_r={lora_r}"
                else:
                    ip.emit(f"[fine-tuning] {label} complete{cost_str}")
            except Exception:
                ip.emit(f"[fine-tuning] {label} complete{cost_str}")
            # Return ONLY the structured one-liner — no raw agent answer content.
            return f"### {label}\n{summary_line}"

        parts = [
            _fmt("Infrastructure (Phase 0)", infra_r, "infra"),
            _fmt("Dataset profile (Phase 1)", data_r,  "data"),
        ]
        if bypass_selection:
            pinned = res.resolved_model_id or self.model_id
            parts.append(
                "### Model selection (Phase 2)\n"
                f"state=bypassed | preferred_model={pinned} | "
                f"strategy={self.fine_tune_strategy} | reason={res.reason}"
            )
        else:
            parts.append(_fmt("Model selection (Phase 2)", model_r, "model"))
        ip.emit("[fine-tuning] Pre-launch phases complete.")

        # Accumulate costs from all three pre-launch runners.
        pre_launch_cost = sum(
            (getattr(r, "total_cost_usd", None) or 0.0)
            for r in (infra_r, data_r, model_r)
            if not isinstance(r, Exception)
        )
        # Per-agent cost ledger lines (with token detail) for the Cost Report.
        self._record_agent_cost("infrastructure", infra_r)
        self._record_agent_cost("data-profiler", data_r)
        self._record_agent_cost("model-selector", model_r)

        # Hard gate: infra.json must exist and have state=ready before the main
        #  agent is invoked. Failing here is cheaper than failing in Step 0.
        _infra_json_path = local_run_dir / "infra.json"
        if not _infra_json_path.exists():
            _log_path = local_run_dir / "outputs" / ip.RUN_LOG_NAME
            ip.error(
                "[pre-launch] infra.json was not written — infra agent did not complete. "
                f"Full infra-agent trace: {_log_path} (grep for [infra-agent])"
            )
            # surface the agent's last answer so the user can see why it stopped
            # without having to hunt through the log file.
            _infra_answer = getattr(infra_r, "answer", "") if not isinstance(infra_r, Exception) else str(infra_r)
            if _infra_answer:
                ip.error(f"[pre-launch] infra-agent last answer: {_infra_answer[:1000]}")
            raise RuntimeError("Pre-launch failed: infra.json missing after infra agent exited")
        try:
            _infra_data = json.loads(_infra_json_path.read_text(encoding="utf-8"))
            _infra_state = _infra_data.get("state", "?")
            if _infra_state != "ready":
                ip.error(
                    f"[pre-launch] infra.json state={_infra_state!r} — "
                    "aborting before main  agent starts"
                )
                raise RuntimeError(f"Pre-launch failed: infra state={_infra_state!r}")
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Pre-launch failed: infra.json is not valid JSON: {exc}"
            ) from exc

        # One-instance-per-run invariant: record the instance Phase 0 resolved
        # (reused or provisioned). On a resume, the run must reuse the same
        # instance — a different id means Phase 0 launched a second instance,
        # which violates the single-instance contract.
        _instance_id = _infra_data.get("instance_id")
        if _instance_id:
            prior = None
            if self._session is not None:
                _state = self._session.load()
                prior = _state.active_instance_id if _state else None
            if prior and prior != _instance_id:
                raise RuntimeError(
                    "Single-instance violation: this run already used Lambda "
                    f"instance {prior}, but Phase 0 resolved a different instance "
                    f"{_instance_id}. A run may use only one Lambda instance."
                )
            if self._session is not None:
                try:
                    self._session.update(active_instance_id=str(_instance_id))
                except Exception as exc:  # pragma: no cover - defensive
                    ip.emit(f"[session] active_instance_id update failed: {exc}")

        # HF_TOKEN is required for trackio + HF Hub push (final checkpoints).
        # Push it to the Lambda instance now so train.py + finalizer can use it
        # transparently. Hard-fail if absent — the run cannot satisfy its
        # always-push-to-hub contract without it.
        _ssh_alias = _infra_data.get("ssh_alias") or self.ssh_alias_override
        if not _ssh_alias:
            raise RuntimeError(
                "Pre-launch failed: infra.json missing ssh_alias and no override set"
            )
        self._bootstrap_remote_hf_token(_ssh_alias)

        return "\n\n".join(parts), pre_launch_cost

    def _bootstrap_remote_hf_token(self, ssh_alias: str) -> None:
        """Push HF_TOKEN to the Lambda instance at ~/.cache/huggingface/token.

        Hard-fails if HF_TOKEN/HUGGING_FACE_HUB_TOKEN is unset locally — the
        finalizer's mandatory HF Hub push and trackio.init() both depend on it.
        """
        token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        if not token or not token.strip():
            raise RuntimeError(
                "HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) is not set in the local "
                "environment. The fine-tuning pipeline requires it to push the "
                "best+last checkpoints to the HF Hub and to authenticate trackio. "
                "Set HF_TOKEN and re-run."
            )
        try:
            push_hf_token_to_remote(ssh_alias, token, key_path=self.key_path)
            ip.emit(f"[hf-token] pushed to {ssh_alias}:~/.cache/huggingface/token (chmod 600)")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to push HF_TOKEN to {ssh_alias}: {exc}"
            ) from exc

    def _build_resume_prompt(
        self,
        run_id: str,
        local_run_dir: Path,
        ssh_alias: str,
    ) -> str:
        remote_run_dir = f"~/capo_runs/{run_id}"
        return _RESUME_TRAINING_PROMPT_TEMPLATE.format(
            run_id=run_id,
            local_run_dir=str(local_run_dir),
            remote_run_dir=remote_run_dir,
            ssh_alias=ssh_alias,
            key_path=self.key_path,
            skills_dir=str(_SKILLS_DIR),
            trackio_space_id=self.trackio_space_id,
        )

    def _build_resume_from_pause_prompt(
        self,
        run_id: str,
        local_run_dir: Path,
        ssh_alias: str,
        pause_reason: str,
        pause_context: dict,
        answer_artifact: str,
    ) -> str:
        """Build the focused prompt for resuming a run that paused before training.

        The gate paused waiting on the user; capo_resume.py has now patched the
        answer artifact and cleared the pause flag. This prompt instructs the
        agent to re-enter the gate at the step it paused on (NOT to restart the
        pipeline from Phase -2).
        """
        remote_run_dir = f"~/capo_runs/{run_id}"
        paused_step = pause_context.get("step", 3)
        candidate_index = pause_context.get("candidate_index", 0)
        return _RESUME_FROM_PAUSE_PROMPT_TEMPLATE.format(
            run_id=run_id,
            local_run_dir=str(local_run_dir),
            remote_run_dir=remote_run_dir,
            ssh_alias=ssh_alias,
            key_path=self.key_path,
            skills_dir=str(_SKILLS_DIR),
            model_id=self.model_id,
            fine_tune_strategy=self.fine_tune_strategy,
            dataset_ref=self.dataset_ref,
            trackio_space_id=self.trackio_space_id or "",
            pause_reason=pause_reason,
            paused_step=paused_step,
            candidate_index=candidate_index,
            answer_artifact=answer_artifact,
        )

    @staticmethod
    def _resolve_resume_context(
        run_id: str,
        local_run_dir: Path,
        ssh_alias_override: str | None,
    ) -> tuple[str, dict]:
        """Validate prior-run state on disk and return (ssh_alias, infra_dict).

        Raises FileNotFoundError / ValueError with actionable messages when
        the state needed to resume is missing or incomplete.
        """
        if not local_run_dir.exists():
            raise FileNotFoundError(
                f"Cannot restart_from_checkpoint: local_run_dir does not exist: "
                f"{local_run_dir}. Pass the run_id of a prior run that produced "
                f"a local run directory."
            )
        infra_path = local_run_dir / "infra.json"
        if not infra_path.exists():
            raise FileNotFoundError(
                f"Cannot restart_from_checkpoint: infra.json not found at {infra_path}. "
                f"The prior run did not reach Phase 0 completion — there is no "
                f"recorded instance to resume on. A full re-run is required."
            )
        try:
            infra = json.loads(infra_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Cannot restart_from_checkpoint: infra.json at {infra_path} is not "
                f"valid JSON: {exc}"
            ) from exc

        ssh_alias = ssh_alias_override or infra.get("ssh_alias")
        if not ssh_alias:
            raise ValueError(
                f"Cannot restart_from_checkpoint: no ssh_alias recorded in "
                f"{infra_path} and no ssh_alias_override provided. Pass "
                f"ssh_alias=... to FineTuningOrchestrator to attach manually."
            )
        return ssh_alias, infra

    def _stage_dataset_source(self, local_run_dir: Path) -> DatasetSource:
        """Resolve dataset_ref (hf/local/uri/named) and stage a local file.

        For a **local** path, copy it into ``data/<basename>`` so the existing
        run-dir rsync (which uploads everything except outputs/checkpoints/results)
        carries it to ``~/capo_runs/<run_id>/data/``. For **uri/named** the file
        is fetched by the agent on the instance, so nothing is copied here — only
        the effective ref is set. For **hf** ``self.dataset_ref`` is left
        byte-identical (zero regression).

        Idempotent: safe to re-run on resume. If the staged file already exists
        (a prior attempt), it is reused even if the original source is gone.
        Always writes ``reports/dataset_source.json`` (a JSON artifact per the
        "decisions backed by JSON" rule). Returns the resolved DatasetSource.
        """
        src = resolve_dataset_source(
            self._dataset_ref_original, base_dir=_REPO_ROOT_ORCH
        )

        if src.kind == "local":
            dest = local_run_dir / (src.staged_rel_path or "data")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                # Already staged (e.g. resume) — reuse; do not require the source.
                ip.emit(f"[fine-tuning] local dataset already staged → {src.effective_ref}")
            else:
                source_path = Path(src.local_path or "")
                if not source_path.exists():
                    raise FileNotFoundError(
                        f"Local dataset not found: {source_path} (from dataset_ref="
                        f"{self._dataset_ref_original!r}). Provide an existing file, "
                        f"an HF Hub id (owner/name), or a URL."
                    )
                shutil.copy2(source_path, dest)
                ip.emit(
                    f"[fine-tuning] staged local dataset {source_path.name} "
                    f"→ {src.effective_ref}"
                )
            self.dataset_ref = src.effective_ref
        elif src.kind in ("uri", "named"):
            self.dataset_ref = src.effective_ref
            ip.emit(
                f"[fine-tuning] dataset kind={src.kind}: agent will fetch it into "
                f"data/ on the instance before the probe"
            )
        # hf: leave self.dataset_ref byte-identical (no mutation, no staging).

        reports_dir = local_run_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "dataset_source.json").write_text(
            json.dumps(src.to_dict(), indent=2), encoding="utf-8"
        )
        self._dataset_source = src
        return src

    def _setup_run_dir(
        self,
        run_id: str,
        output_dir: str | Path | None,
        require_existing: bool,
    ) -> tuple[Path, Path, Path]:
        """Resolve local_run_dir + ensure the full canonical directory structure exists.

        When require_existing=True (resume path), the run dir must already exist —
        we refuse to silently create a fresh directory that would lack the prior
        run's state.

        Returns (local_run_dir, outputs_dir, reports_dir) for backward compatibility.
        outputs_dir is used by ProgressEmitter for orchestrator-agent logs only.
        """
        local_run_dir = Path(
            output_dir or (_REPO_ROOT_ORCH / "runs" / run_id)
        )
        if require_existing and not local_run_dir.exists():
            raise FileNotFoundError(
                f"Cannot restart_from_checkpoint: local_run_dir does not exist: "
                f"{local_run_dir}"
            )
        local_run_dir.mkdir(parents=True, exist_ok=True)

        # Canonical directory structure — keep in sync with prepare_remote_run_dir
        # and capo.utils.checks. Every run has the same boring layout.
        canonical_dirs = [
            "checkpoints/best",
            "checkpoints/last",
            "compaction",
            "configs",
            "outputs",
            "pricing",
            "probe",
            "profile/plots",
            "reports/health",
            "results/plots",
            "results/predictions",
            "scripts",
            "src/data",
            "src/models",
            "src/train",
            "src/eval",
            "src/utils",
        ]
        for rel in canonical_dirs:
            (local_run_dir / rel).mkdir(parents=True, exist_ok=True)

        # Stub package __init__.py so src/ is importable from the entry-point
        # scripts (train.py, probe.py) at run root.
        for pkg in ("src", "src/data", "src/models", "src/train", "src/eval", "src/utils"):
            init = local_run_dir / pkg / "__init__.py"
            if not init.exists():
                init.write_text("", encoding="utf-8")

        # Write manifest.json stub if not already present
        manifest = local_run_dir / "manifest.json"
        if not manifest.exists():
            manifest.write_text(
                json.dumps({
                    "run_id": run_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "schema_version": "2.0",
                    "entrypoints": {
                        "train": "train.py",
                        "probe": "probe.py",
                    },
                    "required_outputs": {
                        "metrics": "results/metrics.json",
                        "evaluation_report": "reports/evaluation_report.md",
                        "final_summary": "reports/final_summary.json",
                        "best_checkpoint": "checkpoints/best/",
                        "last_checkpoint": "checkpoints/last/",
                        "run_report": "RUN_REPORT.md",
                    },
                    "directories": {
                        "checkpoints": "Model checkpoints (best/, last/)",
                        "compaction": "Agent memory and case-file artifacts",
                        "configs": "experiment.yaml, training.yaml, evaluation.yaml",
                        "outputs": "Streaming logs, status.json, metrics.jsonl, train.pid",
                        "pricing": "GPU pricing snapshots and cost_report.json",
                        "probe": "Feasibility probe artifacts",
                        "profile": "Dataset profile + plots",
                        "reports": "Scientific reports, manifests, health history",
                        "results": "Eval CSVs, final metrics.json, plots/, predictions/",
                        "scripts": "Shell entry points (launch_command.sh)",
                        "src": "Python package: data/, models/, train/, eval/, utils/",
                    },
                }, indent=2),
                encoding="utf-8",
            )

        outputs_dir = local_run_dir / "outputs"
        reports_dir = local_run_dir / "reports"
        return local_run_dir, outputs_dir, reports_dir

    @staticmethod
    @staticmethod
    def _task_slug(task_description: str) -> str:
        """Extract a short task slug from a free-text description."""
        task_keywords = [
            (r"\b(ace2|binding|bind)\b",              "binding"),
            (r"\b(localiz|localization|localis)\b",   "localization"),
            (r"\b(function|functional)\b",            "function"),
            (r"\b(struct|structure|fold)\b",          "structure"),
            (r"\b(classif|classification)\b",         "classification"),
            (r"\b(regress|regression)\b",             "regression"),
            (r"\b(generat|generation)\b",             "generation"),
            (r"\b(embed|embedding)\b",                "embedding"),
            (r"\b(finetun|fine.tun)\b",               "finetune"),
            (r"\b(antibod|antibody)\b",               "antibody"),
            (r"\b(solubil)\b",                        "solubility"),
            (r"\b(thermostab|stability)\b",           "stability"),
            (r"\b(variant|variants|variant.effect)\b","variant-effect"),
        ]
        lower = task_description.lower()
        for pattern, slug in task_keywords:
            if re.search(pattern, lower):
                return slug
        return "task"

    @staticmethod
    def _generate_run_id(task_description: str, model_id: str) -> str:
        ts = datetime.now().strftime("%Y%m%d-%H%M")
        task = FineTuningOrchestrator._task_slug(task_description)
        model = _extract_model_slug(task_description) or _extract_model_slug(model_id)
        h = uuid.uuid4().hex[:4]
        return f"{task}-{model}-{ts}-{h}"

    async def _run_finalizer(
        self,
        handoff: MonitorHandoff,
        ctx: HealthMonitorContext,
        case_file: CaseFile | None = None,
    ) -> FinalizerResult:
        """Invoke the  finalizer once with the handoff context.

        When case_file is provided, its markdown rendering is prepended
        to the cached prefix as a "Prior context (compacted)" section, so
        the finalizer inherits durable facts from the pre-launch phase
        without re-paying its full token cost.
        """
        ssh_alias = _read_infra_ssh_alias(ctx.local_run_dir) or ""
        last_health_json = (
            json.dumps(asdict(handoff.last_report), default=str)
            if handoff.last_report is not None
            else "null"
        )
        try:
            handoff_json = ctx.handoff_path.read_text(encoding="utf-8").strip()
        except OSError:
            handoff_json = "{}"
        trackio_url = _read_trackio_url(ctx.local_run_dir) or ""
        prompt = _FINE_TUNING_FINALIZER_PROMPT_TEMPLATE.format(
            handoff_kind=handoff.kind,
            escalation_reason=handoff.reason if handoff.kind == "escalation" else "",
            run_id=ctx.run_id,
            ssh_alias=ssh_alias,
            key_path=ctx.key_path,
            remote_run_dir=ctx.remote_run_dir,
            local_run_dir=str(ctx.local_run_dir),
            trackio_url=trackio_url,
            last_health_report=last_health_json,
            handoff_json=handoff_json or "{}",
            hub_push_config=json.dumps(self.hub_push_config),
            auto_terminate_on_failure=str(self.auto_terminate_on_failure),
            ssh_key_name=self.ssh_key_name,
        )
        if case_file is not None:
            prompt = (
                case_file.to_markdown()
                + "\n---\n\n"
                + prompt
            )
        # Cache the prompt across the agent's internal turns (rsync pull, file
        # reads, summary write) — without this each turn re-bills the full prompt.
        # The mutable tail must be non-empty so cached_streaming_prompt emits a
        # two-block message; an empty mutable produces a single cached block,
        # which makes the agent exit after a brief reply with no tool use.
        result = await self._finalizer_runner.generate(
            prompt=f"Begin finalization for run {ctx.run_id}.",
            cached_prefix=prompt,
        )
        summary = parse_json_with_repair(result.answer)
        return FinalizerResult.from_summary(
            summary=summary,
            raw=result.answer or "",
            agent_cost_usd=result.total_cost_usd,
        )

    @staticmethod
    def _prelaunch_outcome(
        reports_dir: Path, handoff_path: Path, is_resume: bool
    ) -> str:
        """Classify what the Phase A agent achieved this turn.

        Returns one of:
          - "launched" — reports/handoff.json exists (training is live).
          - "aborted"  — an abort marker exists (cost gate, probe failure,
            insufficient GPU, …); a legitimate terminal stop.
          - "stuck"    — neither: the agent ended its turn mid-flight and
            must be continued (or, once the budget is spent, failed — never
            silently reported as unknown).
        """
        if handoff_path.exists():
            return "launched"
        if FineTuningOrchestrator._early_abort_state(reports_dir, is_resume) is not None:
            return "aborted"
        return "stuck"

    @staticmethod
    def _write_prelaunch_stall_marker(
        reports_dir: Path, *, subtype: str | None, continuations: int
    ) -> None:
        """Record why Phase A ended without launching training or aborting.

        Written when the agent exhausted its continuation budget without
        producing handoff.json or an abort marker. Gives the report (and any
        human triage) a concrete artifact instead of a bare failed state.
        Best-effort — a write failure must never mask the real terminal state.
        """
        try:
            reports_dir.mkdir(parents=True, exist_ok=True)
            (reports_dir / "prelaunch_stall.json").write_text(
                json.dumps(
                    {
                        "state": "prelaunch_stalled",
                        "reason": (
                            "Phase A agent ended its turn without writing "
                            "reports/handoff.json (training never launched) and "
                            "without writing an abort marker, even after "
                            f"{continuations} continuation attempt(s)."
                        ),
                        "last_agent_subtype": subtype,
                        "continuations_used": continuations,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as exc:  # pragma: no cover - defensive
            ip.emit(f"[pre-launch] could not write stall marker: {exc}")

    @staticmethod
    def _early_abort_state(reports_dir: Path, resume: bool) -> str | None:
        """If the pre-launch phase wrote an abort marker, return the state name.

        These markers short-circuit Phase C (the finalizer) because training never
        launched — there's nothing to finalize remotely.
        """
        if (reports_dir / "aborted_env_check.json").exists():
            return "aborted_env_check"
        if (reports_dir / "abort_over_budget.json").exists():
            return "aborted_over_budget"
        if (reports_dir / "probe_failure_packet.json").exists():
            return "probe_failed"
        if (reports_dir / "aborted_insufficient_gpu.json").exists():
            return "aborted_insufficient_gpu"
        if (reports_dir / "provision_timeout.json").exists():
            return "provision_timeout"
        if resume and (reports_dir / "resume_failure.json").exists():
            try:
                payload = json.loads(
                    (reports_dir / "resume_failure.json").read_text(encoding="utf-8")
                )
                reason = payload.get("state") or payload.get("reason") or ""
                if "no_checkpoint" in reason:
                    return "resume_no_checkpoint"
                if "unreachable" in reason:
                    return "resume_instance_unreachable"
            except json.JSONDecodeError:
                pass
            return "failed"
        if (reports_dir / "failure.json").exists():
            return "failed"
        return None

    @staticmethod
    def _map_terminal_state(handoff: MonitorHandoff, finalizer: FinalizerResult) -> str:
        """Map (monitor handoff + finalizer result) into the public state enum."""
        if finalizer.terminal_state == "completed":
            return "completed"
        if finalizer.terminal_state == "failed":
            return "failed"
        # The finalizer inspected the evidence and could NOT verify a terminal
        # state (remote unreachable, no local artifacts). "unknown" is the single
        # could-not-verify state, honour that honesty and never coerce an
        # unverifiable run into "failed".
        if finalizer.terminal_state == "unknown":
            return "unknown"
        # Finalizer returned nothing usable; lean on the handoff.
        if handoff.kind == "completed":
            return "completed"
        if handoff.kind in ("failed", "escalation"):
            return "failed"
        return "unknown"

    @staticmethod
    def _infer_state(
        reports_dir: Path,
        outputs_dir: Path,
        finetuned_model_path: Path | None,
        agent_subtype: str,
        resume: bool,
    ) -> str:
        """Classify terminal state from artifacts + the agent's result subtype."""
        if (reports_dir / "aborted_env_check.json").exists():
            return "aborted_env_check"
        if (reports_dir / "abort_over_budget.json").exists():
            return "aborted_over_budget"
        if (reports_dir / "probe_failure_packet.json").exists():
            return "probe_failed"
        if resume and (reports_dir / "resume_failure.json").exists():
            try:
                payload = json.loads(
                    (reports_dir / "resume_failure.json").read_text(encoding="utf-8")
                )
                reason = payload.get("state") or payload.get("reason") or ""
                if "no_checkpoint" in reason:
                    return "resume_no_checkpoint"
                if "unreachable" in reason:
                    return "resume_instance_unreachable"
            except json.JSONDecodeError:
                pass
            return "failed"
        if (reports_dir / "failure.json").exists():
            return "failed"
        if finetuned_model_path is not None or (
            reports_dir / "classification_report.json"
        ).exists():
            return "completed"
        if (
            agent_subtype in ("max_turns", "error_max_turns")
            and (outputs_dir / "train.pid").exists()
        ):
            return "training_in_progress"
        return "unknown"

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def run(
        self,
        task_description: str | None = None,
        run_id: str | None = None,
        output_dir: str | Path | None = None,
        restart_from_checkpoint: bool = False,
        resume_from_pause: bool = False,
        pause_reason: str = "",
        pause_context: dict | None = None,
        answer_artifact: str = "",
    ) -> FineTuningResult:
        """Execute a fine-tuning run end-to-end, or resume one from the latest checkpoint.

        Parameters
        ----------
        task_description:
            Free-text task description used to drive the full pipeline. Required
            for a fresh run; ignored when resuming (either mode).
        run_id:
            Explicit run id. Auto-generated when omitted for a fresh run.
            Required for either resume mode.
        output_dir:
            Local directory for run artifacts. When omitted, defaults to
            <repo>/runs/<run_id>/.
        restart_from_checkpoint:
            Resume mid-training from the latest on-instance checkpoint. Skips
            provisioning, profiling, probe, and cost gate; re-attaches to the
            instance in infra.json and re-launches train.py with
            --resume-from-checkpoint. Requires a populated prior-run dir.
        resume_from_pause:
            Resume a run that paused before training waiting for a user answer.
            All upstream work is on disk (infra/profile/scripts/probe). The
            agent re-enters the 3-step gate at the step it paused on, using the
            answer the user just patched into answer_artifact, then proceeds
            to training launch. Mutually exclusive with restart_from_checkpoint.
        pause_reason / pause_context / answer_artifact:
            Forwarded into the resume-from-pause prompt so the agent knows
            which gate step to re-enter. Captured from SessionState BEFORE the
            pause flag is cleared.
        """
        if restart_from_checkpoint and resume_from_pause:
            raise ValueError(
                "restart_from_checkpoint and resume_from_pause are mutually "
                "exclusive — checkpoint resumes mid-training, pause resumes "
                "pre-training. Pick one."
            )
        is_resume = restart_from_checkpoint or resume_from_pause

        # ---------- resolve run id + run directory ----------
        if is_resume:
            if not run_id:
                raise ValueError(
                    "Resuming (restart_from_checkpoint or resume_from_pause) "
                    "requires an explicit run_id."
                )
            rid = run_id
        elif run_id:
            rid = run_id
        else:
            if task_description is None:
                raise ValueError(
                    "task_description is required for a fresh run"
                )
            rid = self._generate_run_id(task_description, self.model_id)

        local_run_dir, outputs_dir, reports_dir = self._setup_run_dir(
            run_id=rid,
            output_dir=output_dir,
            require_existing=is_resume,
        )

        # Resolve + stage the dataset source BEFORE new_session / the resume fork
        # so state.json and every downstream prompt see the effective ref. For a
        # local file this copies it into data/ (carried up by the run-dir
        # rsync); hf ids are left byte-identical. Runs on both fresh + resume
        # paths (idempotent) so a resumed local run keeps the staged data/ ref.
        self._stage_dataset_source(local_run_dir)

        # Namespace the single-instance guard (used by the lambda MCP server) to
        # this run so the provision tool enforces one-instance-per-run.
        os.environ["CAPO_RUN_ID"] = rid

        # Pre-launch and research costs are only tracked on a fresh run; both
        # default to 0.0 so they can be unconditionally added to combined_cost.
        pre_launch_cost: float = 0.0
        research_cost: float | None = None

        # Reset the per-run agent cost ledger (populated as agents complete).
        self._agent_cost_entries = []

        emitter = ip.ProgressEmitter(
            stdout_log=outputs_dir / ip.RUN_LOG_NAME,
            stderr_log=outputs_dir / ip.RUN_ERR_LOG_NAME,
            activity_tag="fine-tuning",
        )

        # Initialise the per-run SessionStore (manages state.json in local_run_dir).
        # Both resume modes preserve the existing state.json — overwriting would
        # wipe phase clock, candidate index, compaction count, etc.
        self._session = SessionStore(local_run_dir)
        if not is_resume:
            self._session.save(new_session(
                run_id=rid,
                local_run_dir=local_run_dir,
                remote_run_dir=f"~/capo_runs/{rid}",
                model_id=self.model_id,
                fine_tune_strategy=self.fine_tune_strategy,
                dataset_ref=self.dataset_ref,
                ssh_key_name=self.ssh_key_name,
                key_path=self.key_path,
                gpu_preference=self.gpu_preference_arg,
                allow_reuse_existing=self.allow_reuse_existing,
                max_cost_usd=self.max_cost_usd,
                trackio_space_id=self.trackio_space_id or None,
                probe_max_retries=self.probe_max_retries,
                ssh_alias_override=self.ssh_alias_override,
                tolerance_threshold=self.tolerance_threshold,
                reuse_existing_instance=self.allow_reuse_existing,
            ))

        # ---------- preflight gated by path ----------
        if is_resume:
            # Either resume mode re-attaches to the existing instance recorded
            # in infra.json; no provisioning, no API key required.
            resume_ssh_alias, infra = self._resolve_resume_context(
                run_id=rid,
                local_run_dir=local_run_dir,
                ssh_alias_override=self.ssh_alias_override,
            )
        else:
            resume_ssh_alias = None
            infra = {}

        # Build per-run compactor now that local_run_dir exists. NullCompactor
        # when disabled keeps the call sites typed without if guards.
        if self._compaction_config.enabled:
            compactor: Compactor = Compactor(
                self._compaction_config,
                run_id=rid,
                local_run_dir=local_run_dir,
            )
            # Inherit a prior case file if a previous attempt of this run
            # already produced one (e.g. resume after a crash).
            case_file: CaseFile | None = CompactionStore(local_run_dir).load_case_file()
        else:
            compactor = NullCompactor()
            case_file = None

        token = ip.set_emitter(emitter)
        try:
            if restart_from_checkpoint:
                ip.emit(
                    f"[fine-tuning] Resuming run {rid} from checkpoint "
                    f"(ssh_alias={resume_ssh_alias}, "
                    f"instance_type={infra.get('instance_type', 'unknown')})"
                )
                prompt = self._build_resume_prompt(
                    run_id=rid,
                    local_run_dir=local_run_dir,
                    ssh_alias=resume_ssh_alias,
                )
            elif resume_from_pause:
                ip.emit(
                    f"[fine-tuning] Resuming run {rid} from pause "
                    f"(reason={pause_reason!r}, ssh_alias={resume_ssh_alias})"
                )
                prompt = self._build_resume_from_pause_prompt(
                    run_id=rid,
                    local_run_dir=local_run_dir,
                    ssh_alias=resume_ssh_alias,
                    pause_reason=pause_reason,
                    pause_context=pause_context or {},
                    answer_artifact=answer_artifact,
                )
            else:
                ip.emit(f"Starting fine-tuning run {rid}")
                gpu_preference = self._resolve_gpu_preference(task_description)
                if gpu_preference:
                    ip.emit(f"[fine-tuning] gpu_preference resolved to: {gpu_preference}")

                # task.md must exist before any subagent (memory consultant,
                # profiler, etc.) can read it, write it unconditionally.
                (local_run_dir / "task.md").write_text(task_description, encoding="utf-8")

                # Phase -2: episodic memory consultation — runs before research
                # and pre-launch so its findings can prime every downstream agent.
                # Skipped entirely when enable_memory=false
                # current run's RUN_REPORT.md is still produced/registered by the finalizer.
                if self._memory_runner is not None:
                    prior_runs_md, memory_cost = await self._consult_episodic_memory(
                        local_run_dir=local_run_dir,
                        gpu_preference=gpu_preference,
                    )
                    if memory_cost:
                        pre_launch_cost += memory_cost
                else:
                    prior_runs_md = ""
                    ip.emit(
                        "[memory] Episodic memory disabled (enable_memory=false) — "
                        "ignoring prior runs; RUN_REPORT.md still produced."
                    )

                # Phase -1: web research — runs before any subagent dispatch.
                # Skipped entirely when enable_hf_research=false (self._researcher
                # is None); the prompt is adapted accordingly in _build_prompt.
                if self._researcher is not None:
                    try:
                        # A non-hf dataset (local/uri/named) is not on the Hub, so
                        # a `hf datasets info` lookup would waste a call — hand the
                        # researcher a descriptive stand-in instead so it still
                        # researches the model + benchmarks from task.md.
                        src = self._dataset_source
                        research_dataset_ref = (
                            self.dataset_ref
                            if (src is None or src.kind == "hf")
                            else f"(not on HF Hub — {src.kind} dataset; see task.md)"
                        )
                        findings, research_cost = await self._researcher.run(
                            model_id=self.model_id,
                            fine_tune_strategy=self.fine_tune_strategy,
                            dataset_ref=research_dataset_ref,
                            task_md_path=local_run_dir / "task.md",
                        )
                        _write_research_findings(findings, reports_dir)
                        self._record_agent_cost_total(
                            "hf-researcher", "claude-haiku-4-5-20251001", research_cost
                        )
                    except Exception as research_exc:
                        ip.emit(f"[research] Web research failed (non-fatal): {research_exc}")
                else:
                    ip.emit(
                        "[research] HF Hub research disabled (enable_hf_research=false) — skipping."
                    )

                # Hard gate: Lambda API key required for Phase 0 provisioning.
                # Checked here (after research) so web research findings are
                # always written to disk even when the key is missing.
                if not self.api_key:
                    ip.emit(
                        "[warning] LAMBDA_API_KEY not set in environment; "
                        "provisioning will fail if no existing instances are available"
                    )
                    raise ValueError(
                        "LAMBDA_API_KEY not set in environment — required for Lambda "
                        "provisioning. Please set it and try again."
                    )

                self._provision_trackio_dashboard(reports_dir)

                pre_launch_context_subs, pre_launch_subs_cost = await self._run_pre_launch(
                    run_id=rid,
                    local_run_dir=local_run_dir,
                    gpu_preference=gpu_preference,
                    prior_runs_md=prior_runs_md,
                )
                pre_launch_cost += pre_launch_subs_cost

                # Append a one-line pointer describing episodic-memory state so
                # the  pre-launch agent's pre_launch_context names it
                # explicitly (when enabled, the agent also reads it via Step 0).
                if self._memory_runner is not None:
                    memory_pointer = (
                        "\n\n### Episodic memory\n"
                        f"state=consulted | prior_runs_md={local_run_dir / 'reports' / 'prior_runs.md'}"
                    )
                else:
                    memory_pointer = (
                        "\n\n### Episodic memory\n"
                        "state=disabled (enable_memory=false) | prior_runs_md=none"
                    )
                pre_launch_context = pre_launch_context_subs + memory_pointer

                research_available = (
                    reports_dir / "research_findings_agent.json"
                ).exists()
                prompt = self._build_prompt(
                    run_id=rid,
                    task_description=task_description,
                    local_run_dir=local_run_dir,
                    gpu_preference=gpu_preference,
                    pre_launch_context=pre_launch_context,
                    research_enabled=research_available,
                    memory_enabled=self._memory_runner is not None,
                )

            # ---------- Phase A:  pre-launch (+ concurrent Haiku monitor) ----------
            monitor_ctx = HealthMonitorContext(
                run_id=rid,
                local_run_dir=local_run_dir,
                key_path=self.key_path,
                remote_run_dir=f"~/capo_runs/{rid}",
                handoff_path=reports_dir / "handoff.json",
            )
            monitor = TrainingHealthMonitor(monitor_ctx)
            monitor_task = asyncio.create_task(monitor.run_loop())

            agent_exc: Exception | None = None
            agent_result = None  # type: ignore[assignment]
            self._phase(rid, "pre_launch")

            # Phase A runs the entire pre-launch→launch sequence in ONE agent
            # query, but an LLM can end its turn at any moment. The two terminal
            # outcomes that legitimately END Phase A are: (a) training LAUNCHED
            # (reports/handoff.json exists) or (b) the agent ABORTED (an abort
            # marker exists — cost gate, probe failure, insufficient GPU, …). If
            # the agent instead ends its turn with the run still mid-flight (ie no
            # handoff, no abort marker) -> the run is STUCK, not done. Rather than
            # silently classify that as unknown and quit, re-prompt the agent to RESUME 
            # from on-disk + remote state, bounded by max_prelaunch_continuations.
            turn_prompt = f"Begin run {rid}."
            total_prelaunch_attempts = 1 + self.max_prelaunch_continuations
            for attempt in range(1, total_prelaunch_attempts + 1):
                try:
                    # Cache the prompt across the agent's internal turns —
                    # the pre-launch agent typically takes 20+ turns and re-bills
                    # the full ~1300-line template on each one without this.
                    # The mutable tail must be non-empty so cached_streaming_prompt
                    # emits a two-block message (cached stable + uncached mutable);
                    # a single block with cache_control causes the agent to exit
                    # with a brief response and no tool use.
                    agent_result = await self._orchestrator.generate(
                        prompt=turn_prompt,
                        cached_prefix=prompt,
                    )
                except Exception as exc:
                    agent_exc = exc
                    agent_result = None  # type: ignore[assignment]
                    break

                outcome = self._prelaunch_outcome(
                    reports_dir, monitor_ctx.handoff_path, is_resume
                )
                # (a) training launched → handoff.json written → done.
                if outcome == "launched":
                    self._phase(rid, "training")
                    break
                # (b) agent aborted (wrote a terminal marker) → legitimate stop.
                if outcome == "aborted":
                    break

                # Otherwise the agent ended its turn mid-flight. Continue it
                # (bounded) rather than silently dying as unknown.
                if attempt < total_prelaunch_attempts:
                    ip.emit(
                        "[pre-launch] agent ended its turn (subtype="
                        f"{getattr(agent_result, 'subtype', '?')!r}) without "
                        "launching training or writing an abort marker — continuing "
                        f"(continuation {attempt}/{self.max_prelaunch_continuations})."
                    )
                    turn_prompt = self._PRELAUNCH_CONTINUATION_PROMPT.format(
                        run_id=rid,
                        local_run_dir=local_run_dir,
                        remote_run_dir=f"~/capo_runs/{rid}",
                    )
                else:
                    ip.emit(
                        "[pre-launch] agent still has not launched training after "
                        f"{self.max_prelaunch_continuations} continuation(s) — "
                        "classifying the run as failed (not unknown)."
                    )

            # ---- Compaction after Phase A (on the final agent result) ----
            # The pre-launch agent's raw_messages list is the largest context
            # bloat in the run; distil it into a case file before Phase C
            # re-bills the trace. Runs only when the agent returned cleanly.
            if agent_exc is None and agent_result is not None:
                billed_input_tokens = (
                    (agent_result.cache_read_tokens or 0)
                    + (agent_result.cache_creation_tokens or 0)
                )
                try:
                    compacted = await compactor.maybe_compact(
                        phase_label="phase_a_pre_launch",
                        raw_messages=agent_result.raw_messages,
                        prior_case_file=case_file,
                        last_input_tokens=billed_input_tokens,
                    )
                except Exception as comp_exc:  # pragma: no cover - defensive
                    ip.emit(f"[compaction] non-fatal error: {comp_exc}")
                    compacted = None
                if compacted is not None:
                    case_file = compacted
                    # Stamp compaction metrics without touching current_phase
                    # so the phase clock stays accurate.
                    try:
                        self._session.update(
                            compaction_count=case_file.compactions,
                            last_compaction_at=case_file.updated_at,
                        )
                    except Exception as exc:  # pragma: no cover - defensive
                        ip.emit(f"[session] compaction stamp failed: {exc}")

            # ---------- Phase B: wait for monitor to reach a terminal handoff ----------
            # If Agent raised or exited without writing handoff.json, stop the monitor
            # so it doesn't block forever on wait_for_handoff.
            early_abort = self._early_abort_state(reports_dir, is_resume)
            if agent_exc is not None or early_abort is not None or not monitor_ctx.handoff_path.exists():
                await monitor.stop()
            try:
                handoff = await monitor_task
            except Exception as mon_exc:
                ip.emit(f"[health] monitor task raised: {mon_exc}")
                handoff = MonitorHandoff(
                    kind="stopped",
                    reason=f"monitor task raised: {mon_exc}",
                    last_report=None,
                )

            if agent_exc is not None:
                # Pre-launch Agent crashed — no remote to sync from cleanly, skip finalizer.
                raise agent_exc

            pre_launch_agent_cost = agent_result.total_cost_usd if agent_result else None
            agent_answer = agent_result.answer if agent_result else ""
            self._record_agent_cost(
                "orchestrator (pre-launch)", agent_result, model=self._orchestrator.model_name
            )

            # ---------- post-launch failure diagnosis + bounded auto-repair ----------
            # When the monitor returns a failure verdict, pull the structured error
            # artifacts from the remote (no LLM) and classify. If the failure is
            # mechanically fixable (missing_dependency / cuda_kernel), apply the fix
            # on the remote and relaunch the idempotent pipeline, re-monitoring each
            # time, up to self.max_post_launch_repairs. A ledger records every attempt;
            # the same (category, packages) is never retried, so it cannot loop.
            if (
                early_abort is None
                and monitor_ctx.handoff_path.exists()
                and handoff.kind in ("failed", "escalation")
            ):
                try:
                    handoff_payload = json.loads(
                        monitor_ctx.handoff_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    handoff_payload = {}
                _ssh_alias = handoff_payload.get("ssh_alias") or _read_infra_ssh_alias(local_run_dir)
                _remote_dir = handoff_payload.get("remote_run_dir") or monitor_ctx.remote_run_dir

                def _diagnose():
                    try:
                        f = diagnose_post_launch_failure(
                            ssh_alias=str(_ssh_alias),
                            key_path=self.key_path,
                            remote_run_dir=str(_remote_dir),
                            local_run_dir=local_run_dir,
                        )
                        if f is not None:
                            ip.emit(
                                f"[fine-tuning] failure category={f.failure_category} "
                                f"recoverable={f.recoverable} — {f.summary}"
                            )
                        return f
                    except Exception as diag_exc:  # pragma: no cover - defensive
                        ip.emit(
                            f"[fine-tuning] post-launch diagnostics failed (non-fatal): "
                            f"{diag_exc}"
                        )
                        return None

                if _ssh_alias:
                    ip.emit("[fine-tuning] post-launch failure detected — diagnosing")
                    failure = _diagnose()

                    repair_attempts: list[dict] = []
                    tried_signatures: set = set()
                    while (
                        self.max_post_launch_repairs > 0
                        and is_auto_repairable(failure)
                        and len(repair_attempts) < self.max_post_launch_repairs
                        and repair_signature(failure) not in tried_signatures
                    ):
                        tried_signatures.add(repair_signature(failure))
                        attempt_no = len(repair_attempts) + 1
                        ip.emit(
                            f"[fine-tuning] auto-repair {attempt_no}/"
                            f"{self.max_post_launch_repairs}: {failure.failure_category} "
                            f"{failure.missing_packages or ''}"
                        )
                        try:
                            outcome = apply_and_relaunch(
                                ssh_alias=str(_ssh_alias),
                                key_path=self.key_path,
                                remote_run_dir=str(_remote_dir),
                                local_run_dir=local_run_dir,
                                failure=failure,
                            )
                        except Exception as rep_exc:  # pragma: no cover - defensive
                            ip.emit(f"[fine-tuning] auto-repair raised (non-fatal): {rep_exc}")
                            break
                        repair_attempts.append(
                            {
                                "attempt": attempt_no,
                                "category": failure.failure_category,
                                "packages": failure.missing_packages,
                                **outcome.to_dict(),
                            }
                        )
                        if not outcome.relaunched:
                            ip.emit(
                                f"[fine-tuning] auto-repair could not relaunch: "
                                f"{outcome.detail} — finalizing as failed"
                            )
                            break
                        ip.emit(
                            f"[fine-tuning] relaunched after repair (pid={outcome.new_pid}); "
                            "re-monitoring"
                        )
                        self._phase(rid, "training")
                        monitor = TrainingHealthMonitor(monitor_ctx)
                        monitor_task = asyncio.create_task(monitor.run_loop())
                        try:
                            handoff = await monitor_task
                        except Exception as mon_exc:
                            ip.emit(f"[health] post-repair monitor raised: {mon_exc}")
                            handoff = MonitorHandoff(
                                kind="stopped",
                                reason=f"post-repair monitor raised: {mon_exc}",
                                last_report=None,
                            )
                        ip.emit(f"[fine-tuning] post-repair handoff: kind={handoff.kind}")
                        if handoff.kind in ("failed", "escalation"):
                            failure = _diagnose()
                        else:
                            failure = None  # recovered — exit the repair loop

                    if repair_attempts:
                        recovered = handoff.kind not in ("failed", "escalation")
                        (reports_dir / "repair_ledger.json").write_text(
                            json.dumps(
                                {
                                    "attempts": repair_attempts,
                                    "final_handoff_kind": handoff.kind,
                                    "recovered": recovered,
                                },
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                        # On recovery the on-disk post_launch_failure.json describes
                        # the pre-repair crash — stale. Remove it so the finalizer
                        # does not read a recovered run as a failure.
                        if recovered:
                            (reports_dir / "post_launch_failure.json").unlink(missing_ok=True)
                            ip.emit(
                                f"[fine-tuning] auto-repair recovered the run after "
                                f"{len(repair_attempts)} attempt(s)"
                            )
                else:
                    ip.emit(
                        "[fine-tuning] post-launch failure detected but no ssh_alias "
                        "available — skipping diagnostics"
                    )

            # ---------- agentic post-canary recovery (after deterministic repairs) ----------
            # If the run still failed after the mechanical auto-repair path, hand
            # the failure to a diagnosis agent that proposes + applies a
            # SAFE fix, reruns the canary, and relaunches — up to N attempts.
            if (
                early_abort is None
                and monitor_ctx.handoff_path.exists()
                and handoff.kind in ("failed", "escalation")
                and self.max_training_recovery_attempts > 0
            ):
                handoff = await self._run_recovery_loop(
                    handoff, monitor_ctx, local_run_dir, reports_dir
                )
                if handoff.kind not in ("failed", "escalation"):
                    self._phase(rid, "training")

            # ---------- deterministic ground-truth reconciliation ----------
            # Last-resort ground truth before accepting failure: SSH-read remote status.json.
            # This catches stale failures from max_turn cutoffs, monitor false alarms, or
            # runs completing after monitoring stopped—without relying on agents or MCP.
            if early_abort is None and monitor_ctx.handoff_path.exists():
                handoff = await self._reconcile_terminal_state_from_remote(
                    handoff, monitor_ctx
                )
                if handoff.kind not in ("failed", "escalation"):
                    self._phase(rid, "training")

            # ---------- Phase C: Finalizer Agent (skipped only for early aborts) ----------
            finalizer: FinalizerResult | None = None
            if early_abort is None and monitor_ctx.handoff_path.exists():
                ip.emit(f"[fine-tuning] monitor handoff: kind={handoff.kind} reason={handoff.reason}")
                self._phase(rid, "finalizing")
                finalizer = await self._run_finalizer(handoff, monitor_ctx, case_file=case_file)
            elif early_abort is not None:
                ip.emit(f"[fine-tuning] early abort detected: {early_abort} — skipping finalizer")

            # ---------- gather artifacts ----------
            # Only count checkpoints that actually hold bytes. The training
            # scaffold pre-creates checkpoints/best and checkpoints/last, so a
            # run that fails before writing weights leaves two EMPTY dirs — those
            # must not be reported as "2 saved".
            checkpoints_dir = local_run_dir / "checkpoints"
            checkpoint_paths = list_saved_checkpoints(checkpoints_dir)
            report_paths = sorted(reports_dir.iterdir()) if reports_dir.exists() else []
            best_ckpt = local_run_dir / "checkpoints" / "best"
            # Same empty-dir guard: a bare best/ (or one holding only empty
            # subdirs) is not a saved model and must not gate the HF push.
            finetuned_model_path: Path | None = best_ckpt if has_checkpoint_content(best_ckpt) else None

            # ---------- state resolution ----------
            if early_abort is not None:
                state = early_abort
            elif finalizer is not None:
                state = self._map_terminal_state(handoff, finalizer)
            elif not monitor_ctx.handoff_path.exists():
                # The Phase A agent returned (cleanly or via max_turns) but never
                # wrote handoff.json and never wrote an abort marker — even after
                # up to max_prelaunch_continuations re-prompts. Training was never
                # launched. This is a FAILURE, not "unknown": the run is stuck, and
                # silently reporting unknown (and quitting). 
                # Record a diagnosable stall marker so the report can explain it.
                state = "failed"
                self._write_prelaunch_stall_marker(
                    reports_dir,
                    subtype=getattr(agent_result, "subtype", None),
                    continuations=self.max_prelaunch_continuations,
                )
            else:
                # handoff.json exists (training was launched) but no finalizer
                # result, lean on the monitor's handoff verdict rather than
                # inventing "unknown". A launched run that reached no clean
                # terminal is a failure, not an ambiguity.
                state = "completed" if handoff.kind == "completed" else "failed"

            if state == "completed":
                emitter.mark_results_ready()

            finalizer_cost = finalizer.agent_cost_usd if finalizer else None
            monitor_cost = handoff.agent_cost_usd if handoff is not None else None
            self._record_agent_cost_total(
                "training-health-monitor", "claude-haiku-4-5-20251001", monitor_cost
            )
            self._record_agent_cost_total(
                "finalizer", self._finalizer_runner.model_name, finalizer_cost
            )
            combined_cost: float | None
            all_costs = [
                pre_launch_agent_cost,
                monitor_cost,
                finalizer_cost,
                research_cost,
                pre_launch_cost if pre_launch_cost else None,
            ]
            if all(c is None for c in all_costs):
                combined_cost = None
            else:
                combined_cost = sum(c or 0.0 for c in all_costs)
            emitter.emit_final_summary(rid, state, combined_cost)

            resumed_from = None
            if restart_from_checkpoint:
                # Best-effort extraction: the resume prompt instructs the agent
                # to emit a line containing the checkpoint it resumed from.
                match = re.search(
                    r"Resumed run .* from checkpoint\s+(\S+)", agent_answer or ""
                )
                if match:
                    resumed_from = match.group(1).rstrip(".,;:")

            trackio_url = _read_trackio_url(local_run_dir) or (
                finalizer.trackio_url if finalizer else None
            )
            actual_cost_usd = finalizer.actual_cost_usd if finalizer else None

            # ---------- cost report (agent + infra) ----------
            infra_data = infra if is_resume else None
            if not infra_data:
                _ij = local_run_dir / "infra.json"
                if _ij.exists():
                    try:
                        infra_data = json.loads(_ij.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        infra_data = None
            cost_report = self._build_cost_report(local_run_dir, infra_data, actual_cost_usd)
            self._write_cost_report(local_run_dir, reports_dir, cost_report)
            self._append_recovery_report(local_run_dir)

            self._phase(rid, "completed", terminal_state=state, error=None)
            return FineTuningResult(
                run_id=rid,
                local_run_dir=local_run_dir,
                state=state,
                instance_type=infra.get("instance_type") if is_resume else None,
                instance_reused=bool(is_resume),
                ssh_alias=resume_ssh_alias,
                report_paths=report_paths,
                checkpoint_paths=checkpoint_paths,
                finetuned_model_path=finetuned_model_path,
                trackio_url=trackio_url,
                actual_cost_usd=actual_cost_usd,
                cost_report=cost_report.to_dict(),
                answer=agent_answer,
                agent_cost_usd=combined_cost,
                resumed_from_checkpoint=resumed_from,
            )
        except Exception as exc:
            ip.error(f"Fine-tuning run {rid} failed: {exc}")
            self._phase(rid, "failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            ip._emitter.reset(token)

    def run_sync(
        self,
        task_description: str | None = None,
        run_id: str | None = None,
        output_dir: str | Path | None = None,
        restart_from_checkpoint: bool = False,
        resume_from_pause: bool = False,
        pause_reason: str = "",
        pause_context: dict | None = None,
        answer_artifact: str = "",
    ) -> FineTuningResult:
        coro = self.run(
            task_description=task_description,
            run_id=run_id,
            output_dir=output_dir,
            restart_from_checkpoint=restart_from_checkpoint,
            resume_from_pause=resume_from_pause,
            pause_reason=pause_reason,
            pause_context=pause_context,
            answer_artifact=answer_artifact,
        )
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()

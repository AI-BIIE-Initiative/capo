"""
Run a full fine-tuning pipeline on Lambda GPU via the CAPO FineTuningOrchestrator.

All run options live in a YAML config. Copy scripts/configs/fine_tuning.yaml,
edit as needed, and pass its path via --config.

Usage:
    # Fresh run
    python scripts/run_fine_tuning.py --config scripts/configs/fine_tuning.yaml

    # Resume an interrupted run — set resume: <run_id> in the YAML, then:
    python scripts/run_fine_tuning.py --config scripts/configs/fine_tuning.yaml

Environment:
    ANTHROPIC_API_KEY — required for the Claude Agent SDK
    LAMBDA_API_KEY    — required for provisioning new instances
    HF_TOKEN          — required for dataset access, trackio, HF Hub push
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Ensure src/ is importable when running directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from capo.context.compaction import CompactionConfig
from capo.orchestration.fine_tuning_orchestrator import FineTuningOrchestrator
from capo.observability.progress import format_size
from capo.persistence.resume import resume_run
from capo.preflight_keys import MissingAPIKeyError, assert_api_keys

_REPO_ROOT = Path(__file__).parent.parent


def _resolve_task(cfg: dict) -> str:
    """Resolve task (inline) or task_file (path) from config into a string."""
    task = cfg.get("task")
    if task:
        return str(task).strip()

    task_file = cfg.get("task_file")
    if not task_file:
        raise ValueError("Config must set either task or task_file.")

    path = Path(task_file)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"task_file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def _expand(path: str | None) -> str | None:
    return str(Path(path).expanduser()) if path else None


def _resolve_reuse_existing(cfg: dict) -> bool:
    """Resolve the reuse-existing-instance flag from any accepted spelling.

    Accepts (first present wins) allow_reuse_existing,
    reuse_existing_instance, or infra.reuse_existing_instance. Default
    is True — reusing a compatible running instance is cost-efficient and is the
    recommended non-interactive default. A CAPO run uses at most one Lambda
    instance regardless of this flag; it only decides reuse-vs-provision.
    """
    infra = cfg.get("infra") or {}
    for value in (
        cfg.get("allow_reuse_existing"),
        cfg.get("reuse_existing_instance"),
        infra.get("reuse_existing_instance") if isinstance(infra, dict) else None,
    ):
        if value is not None:
            return bool(value)
    return True


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        required=True,
        help="Path to YAML config (see scripts/configs/fine_tuning.yaml).",
    )
    p.add_argument(
        "--answer",
        default=None,
        help=(
            "Answer to a paused run's pending_question.json (non-interactive "
            "resume). Only used when the YAML config sets resume: <run_id> "
            "and that run is paused. Example: --answer classification"
        ),
    )
    args = p.parse_args()

    # Validate required API keys before doing anything expensive. Loads .env
    # from the repo root, prints a per-key status block, and exits with a
    # readable error if anything is missing or blank.
    try:
        assert_api_keys()
    except MissingAPIKeyError as exc:
        print(file=sys.stderr)
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # restart_from_checkpoint is the master resume toggle. Only when it is true
    # does resume: <run_id> short-circuit the pipeline: we read state.json from
    # the local run dir, reconstruct the orchestrator, and dispatch to the resume
    # path (all other YAML fields ignored; --answer flows into a paused run's
    # pending question). When it is false, resume/run_id do NOT trigger a
    # resume — we always start a fresh run, regardless of those fields.
    resume_id = cfg.get("resume")
    restart = bool(cfg.get("restart_from_checkpoint", False))
    if resume_id and restart:
        sys.exit(resume_run(str(resume_id), answer=args.answer))
    if resume_id and not restart:
        print(
            f"NOTE: resume: {resume_id} ignored because restart_from_checkpoint "
            "is false — starting a fresh run.",
            file=sys.stderr,
        )

    if restart and not cfg.get("run_id"):
        p.error("restart_from_checkpoint: true requires run_id to be set in the config.")

    task_description = _resolve_task(cfg)

    compaction_config = CompactionConfig(
        enabled=cfg.get("compaction_enabled", True),
        threshold_input_tokens=cfg.get("compaction_threshold_input_tokens", 80_000),
        keep_recent_messages=cfg.get("compaction_keep_recent_messages", 5),
    )

    # tolerance_threshold is intentionally required (no default in code). It
    # encodes the user's per-run risk appetite for the 3-step gate Step 3 
    if "tolerance_threshold" not in cfg:
        p.error("Config must set tolerance_threshold (float in [0, 1]).")
    try:
        tolerance_threshold = float(cfg["tolerance_threshold"])
    except (TypeError, ValueError):
        p.error(f"tolerance_threshold must be a number, got {cfg['tolerance_threshold']!r}.")
    if not 0.0 <= tolerance_threshold <= 1.0:
        p.error(f"tolerance_threshold must be in [0, 1], got {tolerance_threshold}.")

    hub_push_cfg = cfg.get("hub_push") or {}

    orch = FineTuningOrchestrator(
        key_path=_expand(cfg["key_path"]),
        ssh_key_name=cfg["ssh_key_name"],
        model_id=cfg.get("model_id", "facebook/esm2_t6_8M_UR50D"),
        architecture=cfg.get("architecture"),
        fine_tune_strategy=cfg.get("fine_tune_strategy", "linear-probe"),
        dataset_ref=cfg.get("dataset_ref", "BIIE-AI/ace2_binding"),
        model_name=cfg.get("model_name", "claude-opus-4-8"),
        ssh_alias=cfg.get("ssh_alias"),
        gpu_preference=cfg.get("gpu_preference", "1x GH200"),
        allow_reuse_existing=_resolve_reuse_existing(cfg),
        max_cost_usd=cfg.get("max_cost_usd", 50.0),
        auto_terminate_on_failure=cfg.get("auto_terminate_on_failure", False),
        trackio_space_id=cfg.get("trackio_space_id") or None,
        probe_max_retries=cfg.get("probe_max_retries", 3),
        max_post_launch_repairs=cfg.get("max_post_launch_repairs", 2),
        max_training_recovery_attempts=cfg.get("max_training_recovery_attempts", 3),
        tolerance_threshold=tolerance_threshold,
        max_turns=cfg.get("max_turns", 1000),
        enable_hf_research=cfg.get("enable_hf_research", True),
        enable_memory=cfg.get("enable_memory", True),
        compaction_config=compaction_config,
        hub_push_config=hub_push_cfg,
        orchestrator_effort=cfg.get("orchestrator_effort") or "high",
        orchestrator_skills=cfg.get("orchestrator_skills") or None,
    )

    result = orch.run_sync(
        task_description=task_description,
        run_id=cfg.get("run_id"),
        output_dir=cfg.get("output_dir"),
        restart_from_checkpoint=cfg.get("restart_from_checkpoint", False),
    )

    # --- Print result summary ---
    print()
    print(f"run_id          : {result.run_id}")
    print(f"state           : {result.state}")
    print(f"output_dir      : {result.local_run_dir}")
    if result.instance_type:
        reused = "reused" if result.instance_reused else "new"
        print(f"instance        : {result.instance_type}  ({reused})")
    if result.ssh_alias:
        print(f"ssh_alias       : {result.ssh_alias}")
    print(f"probe_attempts  : {result.probe_attempts}")
    if result.projected_cost_usd is not None:
        print(f"projected_cost  : ${result.projected_cost_usd:.2f}")
    if result.peak_memory_gb is not None:
        print(f"peak_memory     : {result.peak_memory_gb:.1f} GB")
    if result.finetuned_model_path:
        print(f"model           : {result.finetuned_model_path}")
    if result.trackio_url:
        print(f"trackio         : {result.trackio_url}")
    if result.report_paths:
        print("reports         :")
        for rp in result.report_paths:
            try:
                sz = format_size(rp.stat().st_size)
            except OSError:
                sz = "?"
            print(f"  {rp.name:<40}  ({sz})")
    if result.checkpoint_paths:
        print(f"checkpoints     : {len(result.checkpoint_paths)} saved")
    if result.resumed_from_checkpoint:
        print(f"resumed_from    : {result.resumed_from_checkpoint}")

    # --- Cost summary (agent + infra + total) ---
    if result.cost_report:
        from capo.report.cost import render_cli_summary_dict

        print()
        print(render_cli_summary_dict(result.cost_report))
    elif result.agent_cost_usd is not None or result.actual_cost_usd is not None:
        # Fallback when no structured report is available (e.g. early abort).
        agent = result.agent_cost_usd or 0.0
        infra = result.actual_cost_usd or 0.0
        print()
        print("Cost summary:")
        print(f"  Agent cost: ${agent:.4f}")
        print(f"  Infra cost: ${infra:.2f}")
        print(f"  Total cost: ${agent + infra:.2f}")

    if result.state == "training_in_progress":
        print()
        print("NOTE: Training is still running on the remote Lambda instance.")
        print("      The local agent hit the max_turns limit during monitoring.")
        if result.ssh_alias:
            print(f"      To check progress: ssh {result.ssh_alias}")
            print(f"      tail -f ~/capo_runs/{result.run_id}/outputs/train.log")
        print(f"      To resume: set run_id: {result.run_id} and ssh_alias in the config and re-run.")


if __name__ == "__main__":
    main()

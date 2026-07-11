"""
CAPO CLI — entry point (capo).

  capo                  open the interactive CAPO assistant (chat → launch)
  capo --auto           launch non-interactively from config dataset + task
  capo resume <run_id>  resume an interrupted run from its latest checkpoint
  capo health <run_id>  health card for an in-progress or completed run
  capo history          list recent runs as a table
  capo config           interactive arrow-key config editor
  capo prune-memory     interactive memory file cleanup
  capo status <run_id>  one-line status for a run

Bare capo runs a Sonnet-backed assistant that infers the task, asks only the
questions that matter, then builds the SAME FineTuningOrchestrator that
scripts/run_fine_tuning.py builds (from scripts/configs/fine_tuning.yaml). The
chat only shapes the task description fed to task.md — no orchestrator behaviour
changes.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Annotated, Optional

import typer

from capo.observability.progress import RUN_ERR_LOG_NAME, RUN_LOG_NAME

from .colors import console
from .config import CapoConfig, interactive_config_editor, load_config
from .health import print_health_card
from .history import print_history, print_run_detail
from .log_streamer import stream_log
from .logo import print_logo
from .memory import prune_memory
from .questionnaire import RunConfig
from .run_console import (
    print_run_summary,
    prompt_pending_answer,
    read_pending_question,
    run_console,
)

app = typer.Typer(
    name="capo",
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
    invoke_without_command=True,
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _resolve_task_from_config(cfg: CapoConfig) -> Optional[str]:
    """Resolve task (inline) or task_file (path) from config, mirroring
    scripts/run_fine_tuning.py._resolve_task. Returns None if neither is set."""
    if cfg.task:
        return str(cfg.task).strip()
    if cfg.task_file:
        path = Path(cfg.task_file)
        if not path.is_absolute():
            path = cfg.repo_root / path
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    return None


def _build_orchestrator(run: RunConfig):
    """Construct FineTuningOrchestrator from a resolved RunConfig — identical to
    the construction in scripts/run_fine_tuning.py (cwd defaults to repo root so
    .mcp.json is found)."""
    from capo.context.compaction import CompactionConfig
    from capo.orchestration.fine_tuning_orchestrator import FineTuningOrchestrator

    return FineTuningOrchestrator(
        key_path=run.key_path,
        ssh_key_name=run.ssh_key_name,
        model_id=run.model_id,
        fine_tune_strategy=run.fine_tune_strategy,
        dataset_ref=run.dataset_ref,
        tolerance_threshold=run.tolerance_threshold,
        model_name=run.model_name,
        ssh_alias=run.ssh_alias,
        gpu_preference=run.gpu_preference,
        allow_reuse_existing=run.allow_reuse_existing,
        max_cost_usd=run.max_cost_usd,
        trackio_space_id=run.trackio_space_id or None,
        probe_max_retries=run.probe_max_retries,
        max_turns=run.max_turns,
        enable_hf_research=run.enable_hf_research,
        enable_memory=run.enable_memory,
        compaction_config=CompactionConfig(
            enabled=run.compaction_enabled,
            threshold_input_tokens=run.compaction_threshold_input_tokens,
            keep_recent_messages=run.compaction_keep_recent_messages,
        ),
        hub_push_config=run.hub_push,
        orchestrator_effort=run.orchestrator_effort,
        orchestrator_skills=run.orchestrator_skills,
    )


# required keys, in the order shown in the error block (label, env var, example)
_KEY_HINTS = (
    ("Anthropic", "ANTHROPIC_API_KEY", "sk-..."),
    ("Lambda", "LAMBDA_API_KEY", "secret_..."),
    ("HF Token", "HF_TOKEN", "hf_..."),
)


def _print_key_error(missing_labels: set[str]) -> None:
    """Friendly 'set your API keys' block (replaces the raw preflight status dump)."""
    console.print()
    console.print("  [err]CAPO needs your API keys before it can run.[/]")
    console.print()
    for name, var, example in _KEY_HINTS:
        is_set = var not in missing_labels
        mark = "[ok]✓[/]" if is_set else "[err]✗[/]"
        state = "[ok]set[/]" if is_set else "[err]missing[/]"
        console.print(f"    {mark} [accent]{var}[/] [muted]({name})[/] — {state}")
    console.print()
    console.print(
        "  [err]Please set HF_TOKEN, ANTHROPIC_API_KEY and LAMBDA_API_KEY "
        "to continue with capo.[/]"
    )
    console.print()
    console.print(
        "  [muted]Add them to a .env file in the project root, or export them in your shell:[/]"
    )
    for name, var, example in _KEY_HINTS:
        if var in missing_labels:
            console.print(f"    [muted]export {var}={example}[/]")
    console.print()


def _assert_api_keys() -> None:
    """Validate required API keys (loads .env); on failure show the clean error.

    The preflight's own plain-text status block is suppressed (StringIO sink) so
    the CLI shows only the styled block."""
    import io

    from capo.preflight_keys import MissingAPIKeyError, assert_api_keys

    try:
        assert_api_keys(stream=io.StringIO())
    except MissingAPIKeyError as exc:
        _print_key_error({ks.label for ks in exc.missing})
        raise typer.Exit(2)


def _launch(orch, *, task_description, run_id, run_dir: Path, runs_root: Path, from_start: bool,
            interactive: bool = False, cfg: "CapoConfig | None" = None):
    """Wire the live run view + log streamers around orch.run_sync().

    CAPO_PROGRESS_CONSOLE=0 silences the orchestrator's own terminal printing so
    the log streamers (tailing the log files) are the single colour-coded source.

    There is exactly one run view:
      • real terminal → the full-screen RunConsole owns the screen (logs stream in
        a pane, a command bar stays pinned at the bottom). The streamers feed it
        instead of printing and run_sync's stray stdout/stderr is funnelled into
        the log files so nothing else writes the terminal.
      • non-terminal (tests, pipes) or CAPO_RUN_UI=plain → the plain streaming path
        (Rich console.print + a minimal command prompt), behaviour unchanged.
    """
    run_kwargs = dict(
        task_description=task_description, run_id=run_id,
        output_dir=str(run_dir), restart_from_checkpoint=not from_start,
    )
    _run_and_report(orch, run_kwargs, run_id=run_id, run_dir=run_dir,
                    runs_root=runs_root, interactive=interactive, from_start=from_start, cfg=cfg)


def _run_and_report(orch, run_kwargs: dict, *, run_id: str, run_dir: Path, runs_root: Path,
                    interactive: bool, from_start: bool, cfg: "CapoConfig | None" = None) -> None:
    """Drive one orch.run_sync(**run_kwargs) inside the live run view, then print
    the summary. Generic over the run kind: run_kwargs is a fresh-launch call or
    a resume-from-pause call, so an inline answer can re-enter the SAME full-screen
    view (the user asked to 'go back to the full TUI' after answering)."""
    import time as _time

    os.environ["CAPO_PROGRESS_CONSOLE"] = "0"
    outputs = run_dir / "outputs"
    stop = threading.Event()
    # set by /abort from inside the run view. 
    # read after run_sync returns so an abort is detected even when the orchestrator swallows the SIGINT.
    abort_flag = threading.Event()
    use_tui = console.is_terminal and os.environ.get("CAPO_RUN_UI", "").lower() != "plain"

    if use_tui:
        from .run_view import RunConsole

        outputs.mkdir(parents=True, exist_ok=True)  # so the stray-output redirect can open
        run_view = RunConsole(run_id, run_dir, runs_root, stop, _time.monotonic(),
                              abort_event=abort_flag)
        threads = [
            threading.Thread(target=stream_log,
                             args=(outputs / RUN_LOG_NAME, stop, from_start, False, run_view.feed),
                             daemon=True),
            threading.Thread(target=stream_log,
                             args=(outputs / RUN_ERR_LOG_NAME, stop, from_start, True, run_view.feed),
                             daemon=True),
            threading.Thread(target=run_view.run, daemon=True),
        ]
    else:
        threads = [
            threading.Thread(target=stream_log, args=(outputs / RUN_LOG_NAME, stop, from_start),
                             daemon=True),
            threading.Thread(target=stream_log, args=(outputs / RUN_ERR_LOG_NAME, stop, from_start, True),
                             daemon=True),
            threading.Thread(target=run_console, args=(run_id, run_dir, stop, runs_root, abort_flag),
                             daemon=True),
        ]
        console.print()
        console.rule(f"[brand.dim]Run  {run_id}[/]", style="brand.dim")
        console.print()

    for t in threads:
        t.start()

    result = None
    error: Exception | None = None
    aborted = False
    try:
        if use_tui:
            # keep run_sync's stray stdout/stderr off the full-screen alternate
            # screen by routing it into the log files (the streamers tail them into
            # the pane). the orchestrator's own progress already goes there too.
            import contextlib

            with (outputs / RUN_LOG_NAME).open("a", encoding="utf-8", buffering=1) as _o, \
                 (outputs / RUN_ERR_LOG_NAME).open("a", encoding="utf-8", buffering=1) as _e, \
                 contextlib.redirect_stdout(_o), contextlib.redirect_stderr(_e):
                result = orch.run_sync(**run_kwargs)
        else:
            result = orch.run_sync(**run_kwargs)
    except KeyboardInterrupt:
        # /abort or Ctrl+C → the user wants to stop. A confirmed abort from the run
        # view set abort_flag and already announced itself, so only a raw Ctrl+C that
        # bypassed the view's confirmation (early startup / detached mode) needs the
        # notice here — and it is re-confirmed below before anything is torn down.
        aborted = True
        if not abort_flag.is_set():
            console.print("\n  [err]Interrupted[/]  [muted]— run stopped by user.[/]")
    except Exception as exc:  # surface the failure; the orchestrator logs detail
        error = exc
    finally:
        stop.set()
        # in the full-screen path, join the view too so the alternate screen is
        # restored before the summary box prints below- the plain console is a
        # daemon that exits on process end.
        join = threads if use_tui else threads[:2]
        for t in join:
            t.join(timeout=3.0 if use_tui else 1.5)

    # a user abort owns the entire teardown: stop the remote run, sync data.
    aborted = aborted or abort_flag.is_set()
    if aborted:
        if abort_flag.is_set() or _confirm_abort_teardown():
            _abort_cleanup(run_id, run_dir, runs_root, interactive=interactive)
            if interactive and cfg is not None and sys.stdin.isatty():
                _restart_front_door_after_abort(cfg, run_id)
        else:
            console.print(
                "  [muted]Keeping the run as it is — nothing was torn down. "
                "Resume or check it with [/][cmd]capo resume[/][muted] / [/]"
                "[cmd]capo health[/][muted].[/]\n"
            )
        return

    # surface a gate pause (e.g. cost overrun awaiting confirmation); the summary
    # box points the user at capo resume, which asks the pending question and
    # continues. Same facts as before, framed in the brand-dim purple box.
    import json as _json

    try:
        st = _json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    except Exception:
        st = {}
    paused = bool(st.get("paused"))
    # surface the pending question (if any) prominently in the summary box, and —
    # in an interactive terminal — offer to answer it right here. answering only
    # forwards into the existing resume path; the pause/resume contract is intact.
    question = read_pending_question(run_dir, state=st) if paused else None
    print_run_summary(
        run_id,
        run_dir,
        result=result,
        error=error,
        paused=paused,
        pause_reason=st.get("pause_reason"),
        pending_question=question,
    )
    if paused and question and interactive and sys.stdin.isatty():
        _offer_inline_resume(question, run_id=run_id, run_dir=run_dir,
                             runs_root=runs_root, interactive=interactive, cfg=cfg)
    elif (
        interactive
        and not paused
        and cfg is not None
        and sys.stdin.isatty()
        and (result is not None or error is not None)
    ):
        # the run reached a terminal state (not a pause) → drop into the
        # session-aware post-run chat. it can inspect this run's files and, if the
        # user asks to fine-tune again, route back into the same first pipeline.
        _post_run_interaction(cfg, run_id=run_id, run_dir=run_dir, runs_root=runs_root,
                              result=result)


def _post_run_interaction(cfg, *, run_id: str, run_dir: Path, runs_root: Path, result) -> None:
    """Run the post-pipeline chat after a finished run. When the user asks to
    fine-tune again, the chat returns a ChatPlan, which we launch through the SAME
    pipeline used at the start (build orchestrator → live run view) after the same
    budget confirmation. Declining the budget just returns to the chat."""
    from .chat import run_post_run_chat

    while True:
        plan = run_post_run_chat(cfg, runs_root, run_id, run_dir, result=result)
        if plan is None:
            return  # user is done with this run
        if not _confirm_launch(plan):
            console.print(
                "\n  Okay — not launching that. Ask me anything else, or "
                "[cmd]/quit[/] to exit.\n"
            )
            continue  # back to the post-run chat
        # identical seeding to _interactive_launch so the new run mirrors a fresh
        # front-door launch (same run-id slug logic, same enriched task.md).
        import re as _re

        run_cfg = _runconfig_from_plan(cfg, plan)
        model_tokens = _re.sub(r"[/_.\-]", " ", plan.model_id or "")
        seed = f"{plan.objective} {model_tokens}"
        new_run_id, new_run_dir = _resolve_run_ids(cfg, run_cfg, seed)
        _launch(_build_orchestrator(run_cfg), task_description=plan.enriched_description,
                run_id=new_run_id, run_dir=new_run_dir, runs_root=cfg.runs_root,
                from_start=True, interactive=True, cfg=cfg)
        return  # the relaunched run runs its own post-run chat when it finishes


# abort cleanup: stop remote run, sync data, optionally terminate instance 


def _read_remote_run_info(run_id: str, run_dir: Path):
    """Resolve the handles needed to sync + stop an aborted run, or None when the
    run never reached an instance (no infra.json → nothing remote).

    Returns (ssh_alias, key_path|None, remote_run_dir, instance_id, ssh_key_names).
    infra.json carries instance_id + ssh_alias; state.json (the session store)
    carries key_path, ssh_key_name and remote_run_dir."""
    import json as _json

    infra_path = run_dir / "infra.json"
    if not infra_path.exists():
        return None
    try:
        infra = _json.loads(infra_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    instance_id = str(infra.get("instance_id") or "")
    ssh_alias = infra.get("ssh_alias")
    key_path: str | None = None
    ssh_key_names: list[str] = []
    remote_run_dir = f"~/capo_runs/{run_id}"
    try:
        st = _json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        key_path = st.get("key_path") or None
        if st.get("ssh_key_name"):
            ssh_key_names = [st["ssh_key_name"]]
        remote_run_dir = st.get("remote_run_dir") or remote_run_dir
        ssh_alias = ssh_alias or st.get("ssh_alias_override")
    except (OSError, ValueError):
        pass

    if not ssh_alias:
        return None
    return ssh_alias, key_path, remote_run_dir, instance_id, ssh_key_names


def _confirm_abort_teardown() -> bool:
    """Confirm a raw Ctrl+C before tearing the run + GPU down. Default No so an
    accidental interrupt never destroys work. Non-interactive callers (tests,
    pipes) return True — there is no one to ask, so the interrupt stands. The run
    view confirms inline before it ever reaches this point, so this is only the
    fallback for a Ctrl+C that bypassed it (early startup / detached mode)."""
    if not sys.stdin.isatty():
        return True
    from rich.prompt import Confirm

    return Confirm.ask(
        "\n  [accent]Stop this run?[/] [muted]Any training progress since the last "
        "saved checkpoint will be lost.[/]",
        default=False, console=console,
    )


def _verify_instance_down(instance_id: str, *, timeout_s: float = 90.0,
                          poll_s: float = 5.0) -> bool:
    """Poll Lambda until the instance is gone or winding down. Returns True once it
    is no longer active (a 404 / unqueryable instance, or status terminating /
    terminated), False if it is still active after timeout. Never raises — a probe
    error is treated as 'cannot confirm', not a crash."""
    import time as _time

    from capo.remote import lambda_session

    deadline = _time.monotonic() + timeout_s
    while True:
        try:
            inst = lambda_session.get_instance(instance_id)
        except Exception:
            return True  # 404 / no longer queryable → the instance is gone
        if (getattr(inst, "status", "") or "").lower() in ("terminating", "terminated"):
            return True
        if _time.monotonic() >= deadline:
            return False
        _time.sleep(poll_s)


def _abort_cleanup(run_id: str, run_dir: Path, runs_root: Path, *, interactive: bool) -> None:
    """After a user abort, own the whole teardown with clear, step-by-step progress:
    stop the remote training, pull ALL artifacts down, then terminate the GPU
    instance and verify it is no longer running. Best-effort throughout — a run that
    never provisioned an instance, or any SSH / rsync / API error, is reported and
    skipped, never raised: the run is already stopping and cleanup must not crash
    the CLI. (interactive is accepted for call-site symmetry; teardown is the same
    either way — we never leave a GPU billing after an abort.)"""
    _ = interactive
    console.print(f"\n  [err]■ Aborting run[/] [brand.dim]{run_id}[/]")
    info = _read_remote_run_info(run_id, run_dir)
    if info is None:
        console.print(
            "  [muted]No GPU instance was attached to this run — nothing to tear down.[/]\n"
        )
        return
    ssh_alias, key_path, remote_run_dir, instance_id, ssh_key_names = info

    # 1. stop the remote training process so the GPU isn't left working post-abort.
    console.print("  [accent]→ Stopping training on the instance…[/]")
    try:
        from capo.remote.run_manager import stop_remote_run

        stop_remote_run(ssh_alias, run_id, key_path=key_path)
        console.print("  [ok]✓[/] [muted]Training stopped.[/]")
    except Exception as exc:  # best-effort; the sync below is what protects the work
        console.print(f"  [muted]Could not signal stop (continuing anyway): {exc}[/]")

    # 2. pull everything down so no work is lost when the instance is torn down.
    console.print("  [accent]→ Syncing run data off the instance…[/]")
    try:
        from capo.remote.rsync_manager import download_run_outputs

        download_run_outputs(
            ssh_alias, remote_run_dir, run_dir, key_path=key_path,
            subpaths=["outputs/", "results/", "checkpoints/", "reports/", "profile/"],
        )
        console.print(f"  [ok]✓[/] [muted]Data synced to[/] {run_dir}")
    except Exception as exc:
        console.print(
            f"  [err]Sync failed:[/] {exc}  "
            f"[muted](the instance is still up; its data is intact remotely)[/]"
        )

    # 3. terminate the instance and block until it is confirmed down.
    if not instance_id:
        console.print(
            "  [muted]No instance id on record — terminate it from the Lambda "
            "console to avoid charges.[/]\n"
        )
        return
    console.print(f"  [accent]→ Terminating instance[/] [brand.dim]{instance_id}[/][accent]…[/]")
    try:
        from capo.remote.lambda_session import safe_terminate_instance

        safe_terminate_instance(instance_id, ssh_key_names)
    except Exception as exc:
        console.print(
            f"  [err]Could not terminate {instance_id}:[/] {exc}\n"
            f"  [muted]Terminate it from the Lambda console to avoid charges.[/]\n"
        )
        return
    console.print("  [accent]→ Verifying the instance is no longer running…[/]")
    if _verify_instance_down(instance_id):
        console.print(f"  [ok]✓[/] [muted]Instance[/] {instance_id} [muted]is terminated.[/]\n")
    else:
        console.print(
            f"  [accent]Termination requested for {instance_id}[/] "
            f"[muted]— it is still winding down; it will stop shortly. "
            f"Check the Lambda console if unsure.[/]\n"
        )


def _restart_front_door_after_abort(cfg: "CapoConfig", run_id: str) -> None:
    """After an abort + verified teardown, return the user to the very start — the
    front-door chat — with a short context line. Never the post-run chat and never a
    re-confirmation of the aborted run; it is a clean fresh start. show_welcome=False
    suppresses the 'Fully ready / what would you like to train today' banner + command
    list here — redundant right after an abort; the one aborted line below is enough."""
    console.print(
        f"  [brand]Run [/]{run_id}[brand] aborted.[/]  "
        f"[muted]What would you like to do now?[/]\n"
    )
    _interactive_launch(cfg, show_welcome=False)


def _offer_inline_resume(question: dict, *, run_id: str, run_dir: Path, runs_root: Path,
                         interactive: bool, cfg: "CapoConfig | None" = None) -> None:
    """Ask the paused run's question inline; if answered, apply it via the shared
    resume path and re-enter the SAME live run view so the resumed run keeps the
    full TUI. Deferring just points at capo resume. The pause/resume contract is
    unchanged — prepare_pause_resume reuses the same apply-answer + pause-clear."""
    answer = prompt_pending_answer(question)
    if not answer:
        console.print(
            f"\n  [muted]No answer yet — resume later with [/]"
            f"[cmd]capo resume {run_id}[/][muted].[/]\n"
        )
        return

    console.print(
        f"\n  [brand]Resuming[/] [muted]{run_id} with answer[/] "
        f"[brand.dim]{answer}[/][muted] — back to the live view …[/]\n"
    )
    from capo.persistence.resume import prepare_pause_resume, resume_run

    prepared = prepare_pause_resume(run_id, answer)
    if prepared is None:
        # couldn't prepare a TUI re-entry (no state / not paused / unknown target);
        # fall back to the plain shared resume path so the answer still lands.
        resume_run(run_id, answer=answer)
        return
    orch2, run_kwargs2 = prepared
    _run_and_report(orch2, run_kwargs2, run_id=run_id, run_dir=run_dir,
                    runs_root=runs_root, interactive=interactive, from_start=False, cfg=cfg)


# ── bare capo → interactive assistant (or auto) ─────────────────────────────


@app.callback(invoke_without_command=True)
def _default(
    ctx: typer.Context,
    auto: Annotated[
        bool, typer.Option("--auto", help="Skip the chat; use config dataset + task")
    ] = False,
    dataset: Annotated[
        Optional[str], typer.Option("--dataset", help="Dataset ref (auto mode)")
    ] = None,
    task: Annotated[
        Optional[str], typer.Option("--task", "-t", help="Task description (auto mode)")
    ] = None,
    config_path: Annotated[
        Optional[Path], typer.Option("--config", help="Config YAML (default: fine_tuning.yaml)")
    ] = None,
) -> None:
    """capo — open the interactive CAPO assistant (or launch in auto mode)."""
    if ctx.invoked_subcommand is not None:
        return
    cfg = load_config(config_path)
    cfg.runs_root.mkdir(parents=True, exist_ok=True)
    print_logo()
    if auto or cfg.cli_mode == "auto":
        _auto_launch(cfg, dataset or cfg.dataset_ref, task or _resolve_task_from_config(cfg))
    else:
        _interactive_launch(cfg)


def _resolve_run_ids(cfg: CapoConfig, run_cfg: RunConfig, task_description: str):
    """(run_id, run_dir) — explicit from config, else generated from the task."""
    if run_cfg.run_id:
        run_id = run_cfg.run_id
    else:
        from capo.orchestration.fine_tuning_orchestrator import FineTuningOrchestrator

        run_id = FineTuningOrchestrator._generate_run_id(task_description, run_cfg.model_id)
    run_dir = Path(run_cfg.output_dir).expanduser() if run_cfg.output_dir else cfg.runs_root / run_id
    return run_id, run_dir


def _auto_launch(cfg: CapoConfig, dataset_ref: Optional[str], task: Optional[str]) -> None:
    """Non-interactive launch: dataset + task come from flags/config, no chat."""
    if not dataset_ref or not task:
        console.print(
            "  [err]Auto mode needs a dataset ref and a task.[/]\n"
            "  Pass them: [cmd]capo[/] [cmd]--auto --dataset[/] [cmd.arg]BIIE-AI/ace2_binding[/] "
            "[cmd]--task[/] [cmd.arg]'binary binding'[/]\n"
            "  or set dataset_ref + task/task_file in the config.\n"
        )
        raise typer.Exit(1)
    run_cfg = RunConfig.from_config(cfg)
    run_cfg.dataset_ref = dataset_ref
    run_cfg.task_description = task
    console.print(
        f"  [muted]Auto mode — dataset[/] [brand.dim]{dataset_ref}[/]  "
        f"[muted]strategy[/] [brand.dim]{run_cfg.fine_tune_strategy}[/]\n"
    )
    _assert_api_keys()
    run_id, run_dir = _resolve_run_ids(cfg, run_cfg, task)
    # auto mode feeds the orchestrator the raw task (it enriches via task.md).
    _launch(_build_orchestrator(run_cfg), task_description=task, run_id=run_id,
            run_dir=run_dir, runs_root=cfg.runs_root, from_start=True)


def _runconfig_from_plan(cfg: CapoConfig, plan) -> RunConfig:
    rc = RunConfig.from_config(cfg)
    rc.dataset_ref = plan.dataset_ref
    rc.task_description = plan.objective
    rc.fine_tune_strategy = plan.fine_tune_strategy
    rc.max_cost_usd = plan.max_cost_usd
    rc.gpu_preference = plan.gpu_preference
    rc.model_id = plan.model_id or rc.model_id
    return rc


def _confirm_launch(plan) -> bool:
    """Clean pre-launch summary; confirm only the budget (not the whole config)."""
    from rich.panel import Panel
    from rich.prompt import Confirm
    from rich.table import Table

    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column("Key", style="metric.key", no_wrap=True)
    t.add_column("Value", style="brand.dim")
    rows = [
        ("Task", plan.objective or "—"),
        ("Mode", plan.mode),
        ("Dataset", plan.dataset_ref),
        ("Model", plan.model_id or "auto-select"),
        ("Strategy", plan.fine_tune_strategy),
        ("GPU", plan.gpu_preference or "auto"),
        ("Budget", f"${plan.max_cost_usd:.0f}"),
    ]
    for k, v in rows:
        t.add_row(k, v)
    console.print()
    console.print(Panel(t, title="[brand] Ready to launch[/]", border_style="brand.dim",
                        padding=(0, 1)))
    return Confirm.ask(
        f"\n  [brand]Start this run with a budget of ${plan.max_cost_usd:.0f}?[/]",
        default=True, console=console,
    )


def _interactive_launch(cfg: CapoConfig, *, show_welcome: bool = True) -> None:
    """The bare-capo experience: chat → confirm budget → launch. show_welcome=False
    skips the opening 'Fully ready' banner — used when re-entering after an abort,
    where the caller has already printed its own one-line context."""
    from .chat import run_chat

    _assert_api_keys()  # the chat needs the Anthropic key — fail clean before chatting
    plan = run_chat(cfg, cfg.runs_root, show_welcome=show_welcome)
    if plan is None:
        return  # user quit — goodbye already printed
    if not _confirm_launch(plan):
        console.print(
            "\n  Okay. To change the configuration, type [cmd]/config[/] "
            "and edit the values you want.\n"
        )
        raise typer.Exit(0)
    run_cfg = _runconfig_from_plan(cfg, plan)
    # the task.md keeps the full HF id (e.g. esm2_t12_35M_UR50D); the run-id slug
    # extractor needs the model family as a standalone token, so seed it with a
    # separator-tokenised objective + model id rather than the whole brief.
    import re as _re

    model_tokens = _re.sub(r"[/_.\-]", " ", plan.model_id or "")
    seed = f"{plan.objective} {model_tokens}"
    run_id, run_dir = _resolve_run_ids(cfg, run_cfg, seed)
    _launch(_build_orchestrator(run_cfg), task_description=plan.enriched_description,
            run_id=run_id, run_dir=run_dir, runs_root=cfg.runs_root, from_start=True,
            interactive=True, cfg=cfg)


# ── capo resume ──────────────────────────────────────────────────────────────


@app.command()
def resume(
    run_id: Annotated[str, typer.Argument(help="Run ID to resume")],
    answer: Annotated[
        Optional[str],
        typer.Option("--answer", help="Answer to a paused run's pending question, e.g. 'accept'"),
    ] = None,
) -> None:
    """Resume an interrupted run.

    Delegates to the shared resume path (capo.persistence.resume.resume_run),
    which handles both cases from state.json:
      - paused for user input (e.g. cost-overrun confirmation) → asks the
        pending question (use --answer to answer non-interactively), patches the
        artifact, then re-enters the gate;
      - interrupted mid-training → resumes from the latest on-instance checkpoint.
    """
    from capo.persistence.resume import resume_run

    print_logo()
    raise typer.Exit(resume_run(run_id, answer=answer))


# ── capo config ──────────────────────────────────────────────────────────────


@app.command(name="config")
def config_cmd(
    config_path: Annotated[Optional[Path], typer.Option("--config")] = None,
) -> None:
    """Interactive arrow-key config editor (edit values, write back)."""
    print_logo()
    interactive_config_editor(load_config(config_path))


# ── capo health ──────────────────────────────────────────────────────────────


@app.command()
def health(
    run_id: Annotated[str, typer.Argument(help="Run ID")],
    runs_dir: Annotated[Optional[Path], typer.Option("--runs-dir")] = None,
    watch: Annotated[bool, typer.Option("--watch", "-w", help="Refresh every 10s")] = False,
) -> None:
    """Show health metrics for a run (loss, MCC/AUROC, GPU, cost, trackio)."""
    import time

    runs_root = runs_dir or load_config().runs_root
    if not watch:
        print_health_card(run_id, runs_root)
        return
    try:
        while True:
            console.clear()
            print_health_card(run_id, runs_root)
            console.print("  [muted]Refreshing every 10s — Ctrl+C to quit[/]\n")
            time.sleep(10)
    except KeyboardInterrupt:
        pass


# ── capo history ─────────────────────────────────────────────────────────────


@app.command()
def history(
    runs_dir: Annotated[Optional[Path], typer.Option("--runs-dir")] = None,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
    full: Annotated[
        Optional[str],
        typer.Option("--full", help="Show full, untruncated detail for one RUN_ID"),
    ] = None,
) -> None:
    """List recent fine-tuning runs (or full detail for one with --full)."""
    runs_root = runs_dir or load_config().runs_root
    if full:
        print_run_detail(runs_root, full)
    else:
        print_history(runs_root, limit=limit)


# ── capo inspect ─────────────────────────────────────────────────────────────


@app.command()
def inspect(
    run_id: Annotated[str, typer.Argument(help="Run ID to inspect")],
    runs_dir: Annotated[Optional[Path], typer.Option("--runs-dir")] = None,
) -> None:
    """Full detail for one run plus the artifacts present in its run directory."""
    print_run_detail(runs_dir or load_config().runs_root, run_id, list_artifacts=True)


# ── capo prune-memory ────────────────────────────────────────────────────────


@app.command(name="prune-memory")
def prune_memory_cmd(
    run_id: Annotated[
        Optional[str], typer.Argument(help="Run ID to forget (omit for an interactive picker)")
    ] = None,
    index_path: Annotated[
        Optional[Path], typer.Option("--index", help="Path to runs_index.md")
    ] = None,
) -> None:
    """Forget a run from CAPO memory (removes it from runs_index.md only).

    Stops the memory consultant from rediscovering the run; the run directory,
    RUN_REPORT.md, checkpoints and artifacts are never touched. With no RUN_ID an
    interactive multi-select picker opens.
    """
    print_logo()
    index = index_path or load_config().repo_root / "runs" / "runs_index.md"
    prune_memory(index, run_id=run_id)


# ── capo status ──────────────────────────────────────────────────────────────


@app.command()
def status(
    run_id: Annotated[str, typer.Argument(help="Run ID")],
    runs_dir: Annotated[Optional[Path], typer.Option("--runs-dir")] = None,
) -> None:
    """One-line status for a run."""
    import json

    runs_root = runs_dir or load_config().runs_root
    state_path = runs_root / run_id / "state.json"
    if not state_path.exists():
        console.print(f"[err]Not found:[/] {run_id}")
        raise typer.Exit(1)
    d = json.loads(state_path.read_text(encoding="utf-8"))
    phase = d.get("current_phase", "?")
    terminal = d.get("terminal_state", "")
    updated = (d.get("updated_at", "") or "")[:19]
    style = (
        "phase.done" if phase == "completed" else "phase.fail" if phase == "failed" else "phase.run"
    )
    console.print(
        f"  [{style}]{phase}[/]"
        + (f"  [muted]({terminal})[/]" if terminal else "")
        + f"  [muted]{updated}[/]"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()

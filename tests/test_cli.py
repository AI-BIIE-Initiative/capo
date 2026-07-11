"""
Tests for the capo CLI presentation layer (src/capo/cli).

Covers the pure logic (colours, log rendering, config, intent parsing, run-plan
enrichment, slash-command lexer, health/history readers) and the run-command
wiring with a stubbed orchestrator.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
import yaml

# === colours / logo ===


def test_colors_console_and_theme():
    from capo.cli.colors import console, THEME

    assert "cmd" in THEME.styles  # purple slash-command token exists
    assert console is not None


def test_logo_prints(capsys):
    from capo.cli.logo import print_logo

    print_logo()
    out = capsys.readouterr().out
    assert "Compute-Aware Automated Protein Optimization" in out
    assert "CAPO CLI" in out


# === log streamer ===


def test_tag_style_mapping():
    from capo.cli.log_streamer import _tag_style

    assert _tag_style("fine-tuning") == "tag.training"
    assert _tag_style("lambda") == "tag.lambda"
    assert _tag_style("ssh") == "tag.lambda"
    assert _tag_style("rsync") == "tag.rsync"
    assert _tag_style("summary") == "tag.summary"
    assert _tag_style("ssh.cmd") == "tag.default"  # .cmd echoes are dimmed
    assert _tag_style("rsync.cmd") == "tag.default"
    assert _tag_style("totally-unknown") == "tag.default"


def test_render_line_parses_and_passes_through():
    from capo.cli.log_streamer import render_line

    styled = render_line("20:21:05 [fine-tuning]  Training started")
    assert "Training started" in styled.plain
    # a malformed line is returned verbatim (muted)
    assert render_line("no timestamp here").plain == "no timestamp here"


def test_stream_log_tails_new_lines(tmp_path):
    from capo.cli.log_streamer import stream_log

    log = tmp_path / "stdout.log"
    log.write_text("20:00:00 [setup] preexisting\n")  # must NOT be replayed (from_start=False)
    stop = threading.Event()
    t = threading.Thread(target=stream_log, args=(log, stop), daemon=True)
    t.start()
    time.sleep(0.3)
    with log.open("a") as f:
        f.write("20:00:01 [fine-tuning] new line\n")
    time.sleep(0.3)
    stop.set()
    t.join(timeout=2)
    assert not t.is_alive()


# === config ===


def _write_cfg(tmp_path: Path, **overrides) -> Path:
    base = {
        "key_path": "~/.ssh/k",
        "ssh_key_name": "mykey",
        "gpu_preference": "1x A100",
        "allow_reuse_existing": False,
        "model_id": "facebook/esm2_t6_8M_UR50D",
        "fine_tune_strategy": "linear-probe",
        "dataset_ref": "BIIE-AI/ace2_binding",
        "max_cost_usd": 50.0,
        "tolerance_threshold": 0.1,
        "trackio_space_id": "x/y",
        "hub_push": {"enabled": True, "private": True},
        "model_name": "claude-sonnet-4-6",
        "max_turns": 1000,
        "enable_hf_research": True,
        "enable_memory": False,
        "compaction_enabled": True,
        "cli_mode": "interactive",
        # a comment-bearing field to verify set_cli_mode preserves comments
    }
    base.update(overrides)
    text = "# top comment\n" + yaml.safe_dump(base, sort_keys=False) + "\n# trailing comment\n"
    p = tmp_path / "fine_tuning.yaml"
    p.write_text(text)
    return p


def test_load_config_fields(tmp_path):
    from capo.cli.config import load_config

    cfg = load_config(_write_cfg(tmp_path))
    assert cfg.cli_mode == "interactive"
    assert cfg.tolerance_threshold == 0.1
    assert cfg.enable_memory is False
    assert cfg.hub_push == {"enabled": True, "private": True}
    assert cfg.runs_root == cfg.repo_root / "runs"


def test_set_cli_mode_preserves_comments(tmp_path):
    from capo.cli.config import load_config, set_cli_mode

    p = _write_cfg(tmp_path)
    before = sum(1 for ln in p.read_text().splitlines() if ln.strip().startswith("#"))
    set_cli_mode(p, "auto")
    after = p.read_text()
    assert yaml.safe_load(after)["cli_mode"] == "auto"
    assert sum(1 for ln in after.splitlines() if ln.strip().startswith("#")) == before
    assert load_config(p).cli_mode == "auto"


def test_runconfig_from_config_carries_all_fields(tmp_path):
    from capo.cli.config import load_config
    from capo.cli.questionnaire import RunConfig

    cfg = load_config(_write_cfg(tmp_path))
    rc = RunConfig.from_config(cfg)
    assert rc.tolerance_threshold == cfg.tolerance_threshold
    assert rc.enable_memory == cfg.enable_memory
    assert rc.model_name == cfg.model_name
    assert rc.hub_push == cfg.hub_push
    assert rc.max_turns == cfg.max_turns


# === intent prompt ===─


def test_parse_intent_extracts_hints():
    from capo.cli.intent_prompt import parse_intent

    h = parse_intent("ACE2 binding from BIIE-AI/ace2_binding, lora, ESM2, $20 on A100")
    assert h.dataset_ref == "BIIE-AI/ace2_binding"
    assert h.fine_tune_strategy == "lora"
    assert h.max_cost_usd == 20.0
    assert h.gpu_preference == "A100"
    assert h.model_hint == "ESM2"


def test_parse_intent_captures_local_paths_and_urls():
    from capo.cli.intent_prompt import parse_intent

    # Local paths and fetch URLs are now captured verbatim into dataset_ref; the
    # orchestrator's resolve_dataset_source() classifies them (local/uri) later.
    assert parse_intent("train on ~/data/foo with full fine-tuning").dataset_ref == "~/data/foo"
    assert parse_intent("use ./sets/assay.csv").dataset_ref == "./sets/assay.csv"
    assert parse_intent("pull https://ex.com/d.parquet now.").dataset_ref == "https://ex.com/d.parquet"
    # A bare owner/name HF id still parses as before (no path/URL present).
    assert parse_intent("classify binding on owner/ds").dataset_ref == "owner/ds"


def test_parse_intent_hf_and_path_are_equal_precedence():
    from capo.cli.intent_prompt import parse_intent

    # An HF owner/name id is as strong as a local path / URL — whichever is
    # mentioned FIRST wins; neither kind is privileged.
    assert parse_intent("use my-org/binding-set then ./local.csv").dataset_ref == "my-org/binding-set"
    assert parse_intent("./local.csv or my-org/binding-set").dataset_ref == "./local.csv"
    # An owner/name that is obviously a MODEL checkpoint defers to a real dataset
    # signal even when it appears first...
    assert (
        parse_intent("fine-tune facebook/esm2_t6_8M_UR50D on ./data/assay.csv").dataset_ref
        == "./data/assay.csv"
    )
    # ...but a lone model-looking id is still kept (never silently dropped).
    assert parse_intent("train on proteinea/fluorescence").dataset_ref == "proteinea/fluorescence"


# === run planner ======─


def test_infer_modality():
    from capo.cli.run_planner import _infer_modality

    assert _infer_modality("", "protein binding with esm") == "protein_sequence"
    assert _infer_modality("", "scRNA single-cell clustering") == "single_cell"
    assert _infer_modality("", "text sentiment bert") == "text"
    assert _infer_modality("", "tabular csv features") == "tabular"
    assert _infer_modality("", "mystery") == "unknown"


def test_build_run_intent_seen_and_unseen(tmp_path):
    from capo.cli.run_planner import build_run_intent

    # synthesize a prior run for the dataset
    rd = tmp_path / "prior-run"
    rd.mkdir()
    (rd / "state.json").write_text(
        json.dumps({"run_id": "prior-run", "dataset_ref": "owner/ds", "current_phase": "failed"})
    )

    seen = build_run_intent("owner/ds", "binary binding", "lora", 20.0, "A100", None, tmp_path)
    assert seen.seen_before is True and seen.prior_run_id == "prior-run"
    assert seen.inferred_modality == "protein_sequence"
    assert "data modality: protein sequences" in seen.enriched_description
    assert "budget ceiling: $20" in seen.enriched_description

    unseen = build_run_intent("acme/new", "scRNA clustering", "full", 5.0, None, None, tmp_path)
    assert unseen.seen_before is False and unseen.inferred_modality == "single_cell"


def test_build_run_intent_respects_explicit_model_hint(tmp_path):
    from capo.cli.run_planner import build_run_intent

    i = build_run_intent("owner/ds", "binding", "lora", 10.0, None, "facebook/esm2", tmp_path)
    assert "suggested architecture: facebook/esm2" in i.enriched_description


# === run console (lexer / toolbar / notes) ===


def test_slash_lexer_purple_when_resolvable():
    from prompt_toolkit.document import Document

    from capo.cli.run_console import SlashCommandLexer

    lex = SlashCommandLexer()

    def frags(line):
        return lex.lex_document(Document(line))(0)

    # a unique prefix now resolves to one command, so it turns purple too.
    assert any(cls == "class:cmd" and txt == "/heal" for cls, txt in frags("/heal"))
    assert any(cls == "class:cmd" and txt == "/health" for cls, txt in frags("/health"))
    assert any(cls == "class:cmd" and txt == "/tune" for cls, txt in frags("/tune note"))
    # an ambiguous prefix stays white (could still mean several commands).
    assert all(cls == "" for cls, _ in frags("/he"))
    assert all(cls == "" for cls, _ in frags("plain text"))


def test_run_command_bar_and_banner(capsys):
    from prompt_toolkit.formatted_text import to_formatted_text

    from capo.cli.run_console import _PLACEHOLDER, fmt_elapsed, print_run_banner

    assert fmt_elapsed(3725) == "01:02:05"
    # the minimal command bar = a dim placeholder listing the slash commands
    text = "".join(t for _s, t in to_formatted_text(_PLACEHOLDER))
    assert "/health" in text and "/abort" in text and "/history" in text
    # the live banner shows immediately so the run view never looks like a stuck
    # editor waiting for input (no Ctrl+D needed to reach the run layout).
    print_run_banner("binding-esm2-x")
    out = capsys.readouterr().out
    assert "live" in out and "streaming run logs" in out


def test_save_note(tmp_path):
    from capo.cli.run_console import save_note

    save_note(tmp_path, "increase eval freq")
    assert "increase eval freq" in (tmp_path / "user_notes.txt").read_text()


# === health / history readers ===


def _synth_run(root: Path, run_id: str, *, phase="completed", terminal="completed") -> Path:
    rd = root / run_id
    (rd / "outputs").mkdir(parents=True)
    (rd / "reports" / "health").mkdir(parents=True)
    (rd / "pricing").mkdir(parents=True)
    (rd / "state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": "2026-06-16T18:35:00Z",
                "current_phase": phase,
                "terminal_state": terminal,
                "dataset_ref": "BIIE-AI/ace2_binding",
                "model_id": "facebook/esm2_t6_8M_UR50D",
                "fine_tune_strategy": "linear-probe",
            }
        )
    )
    (rd / "outputs" / "status.json").write_text(json.dumps({"state": "completed", "step": 350}))
    (rd / "reports" / "health" / "history.jsonl").write_text(
        json.dumps(
            {
                "state": "completed",
                "step": 350,
                "metrics": {"train_loss": 0.54, "val_loss": 0.48, "val_mcc": 0.92, "val_auroc": 1.0},
                "trend": "improving",
                "severity": "info",
                "gpu_util_pct": 0,
                "gpu_mem_pct": 0,
                "summary": "done",
            }
        )
        + "\n"
    )
    (rd / "reports" / "final_summary.json").write_text(
        json.dumps(
            {
                "terminal_state": "completed",
                "final_metrics": {"test_mcc": 0.69, "test_auroc": 0.96, "val_loss": 0.48},
                "actual_cost_usd": 0.27,
            }
        )
    )
    (rd / "pricing" / "cost_report.json").write_text(
        json.dumps({"training_plan": {"total_steps": 700}, "projections": {"projected_cost_usd": 0.02}})
    )
    return rd


def test_health_card_renders_real_schema(tmp_path, capsys):
    from capo.cli.health import print_health_card

    _synth_run(tmp_path, "r1")
    print_health_card("r1", tmp_path)
    out = capsys.readouterr().out
    assert "Training Health" in out
    assert "350 / 700" in out  # progress from cost_report total_steps
    assert "0.27" in out  # actual cost
    assert "Test MCC" in out


def test_health_card_not_found(tmp_path, capsys):
    from capo.cli.health import print_health_card

    print_health_card("nope", tmp_path)
    assert "Run not found" in capsys.readouterr().out


def test_history_table(tmp_path, capsys, monkeypatch):
    from capo.cli.history import print_history

    monkeypatch.setenv("COLUMNS", "200")  # avoid Rich column truncation under capsys
    _synth_run(tmp_path, "r1")
    print_history(tmp_path, limit=5)
    out = capsys.readouterr().out
    assert "r1" in out and "completed" in out and "0.27" in out


def test_ellipsize():
    from capo.cli.history import ellipsize

    assert ellipsize("abc", 10) == "abc"               # short → unchanged
    assert ellipsize("abcdefghij", 5) == "abcd…"        # truncated, marked, exact width
    assert len(ellipsize("abcdefghij", 5)) == 5
    assert ellipsize(None, 5) == ""                     # tolerant of None
    assert ellipsize("xx", 0) == "xx"                   # non-positive cap → no-op


def _render_at_width(fn, width: int) -> str:
    """Run fn() with the shared history console swapped for a fixed-width recorder
    and return the rendered plain text (so we can measure the layout deterministically)."""
    from rich.console import Console

    import capo.cli.history as h
    from capo.cli.colors import THEME

    rec = Console(width=width, theme=THEME, highlight=False, record=True)
    orig = h.console
    h.console = rec
    try:
        fn()
    finally:
        h.console = orig
    return rec.export_text()


def test_history_table_never_overflows_with_long_dataset(tmp_path):
    from capo.cli.history import print_history

    rd = _synth_run(tmp_path, "esm2-ace2-binding-9fa3")
    # a pathological dataset ref that used to widen its column off-screen.
    state = json.loads((rd / "state.json").read_text())
    state["dataset_ref"] = "Org/" + "x" * 200
    state["model_id"] = "facebook/" + "y" * 120
    (rd / "state.json").write_text(json.dumps(state))

    text = _render_at_width(lambda: print_history(tmp_path), width=100)
    assert max(len(line) for line in text.splitlines()) <= 100  # fits the terminal
    assert "…" in text                                          # something got ellipsized
    assert "x" * 200 not in text                                # the long ref is truncated


def test_print_run_detail_shows_full_values_and_artifacts(tmp_path):
    from capo.cli.history import print_run_detail

    rd = _synth_run(tmp_path, "r1")
    long_ref = "Org/" + "z" * 80
    state = json.loads((rd / "state.json").read_text())
    state["dataset_ref"] = long_ref
    (rd / "state.json").write_text(json.dumps(state))
    (rd / "task.md").write_text("# Task")

    text = _render_at_width(
        lambda: print_run_detail(tmp_path, "r1", list_artifacts=True), width=120
    )
    assert "z" * 80 in text.replace("\n", "").replace(" ", "")  # full ref present (folded, not cut)
    assert "Artifacts" in text and "task brief" in text         # inspect artifact list
    assert "final summary" in text                              # known artifact label shown


def test_print_run_detail_missing_run(tmp_path):
    from capo.cli.history import print_run_detail

    text = _render_at_width(lambda: print_run_detail(tmp_path, "nope"), width=100)
    assert "Not found" in text


# === progress.py quiet flag (surgical change 1) ============


def test_progress_quiet_flag(tmp_path, capsys, monkeypatch):
    from capo.observability.progress import ProgressEmitter

    monkeypatch.delenv("CAPO_PROGRESS_CONSOLE", raising=False)
    ProgressEmitter(stdout_log=tmp_path / "a.log").emit("default-prints")
    assert "default-prints" in capsys.readouterr().out  # default unchanged

    ProgressEmitter(stdout_log=tmp_path / "b.log", console=False).emit("quiet")
    assert "quiet" not in capsys.readouterr().out
    assert "quiet" in (tmp_path / "b.log").read_text()  # still logged

    monkeypatch.setenv("CAPO_PROGRESS_CONSOLE", "0")
    ProgressEmitter(stdout_log=tmp_path / "c.log").emit("env-quiet")
    assert "env-quiet" not in capsys.readouterr().out
    assert "env-quiet" in (tmp_path / "c.log").read_text()


# === hf_research.py (surgical change 2) ============


def test_hf_research_summarise_task_removed():
    from capo.research.hf_research import HFResearcher

    assert not hasattr(HFResearcher, "_summarise_task")


def test_hf_research_run_signature_uses_task_md_path():
    import inspect

    from capo.research.hf_research import HFResearcher

    params = list(inspect.signature(HFResearcher.run).parameters)
    assert "task_md_path" in params and "task_description" not in params


def test_research_user_prompt_accepts_task_context():
    from capo.utils.prompts import load_prompt

    out = load_prompt("research/user_prompts/hf_research").format(
        model_id="m", fine_tune_strategy="lora", dataset_ref="d/x", task_context="READ task.md"
    )
    assert "READ task.md" in out


# === app wiring (auto + interactive) with a stubbed orchestrator =========─


class _FakeResult:
    state = "completed"
    finetuned_model_path = "checkpoints/best/"
    trackio_url = "https://trackio/x"


class _FakeOrch:
    def __init__(self, captured):
        self._c = captured

    def run_sync(self, task_description, run_id, output_dir, restart_from_checkpoint=False):
        self._c["task"] = task_description
        self._c["run_id"] = run_id
        self._c["restart"] = restart_from_checkpoint
        od = Path(output_dir)
        (od / "outputs").mkdir(parents=True, exist_ok=True)
        (od / "state.json").write_text(json.dumps({"current_phase": "training"}))
        (od / "outputs" / "run.log").write_text("20:00:00 [fine-tuning] step 0\n")
        return _FakeResult()


def _auto_cfg(tmp_path: Path) -> Path:
    return _write_cfg(tmp_path, cli_mode="auto", run_id="t-stub", output_dir=str(tmp_path / "t-stub"))


def test_auto_launch_no_prompts(tmp_path, capsys, monkeypatch):
    import capo.cli.app as app
    from capo.cli.config import load_config

    cfg = load_config(_auto_cfg(tmp_path))
    captured: dict = {}
    monkeypatch.setattr(app, "_build_orchestrator", lambda run: _FakeOrch(captured))
    monkeypatch.setattr(app, "_assert_api_keys", lambda: None)
    monkeypatch.setattr(app, "run_console", lambda run_id, run_dir, stop, runs_root, abort_event=None: stop.wait())

    app._auto_launch(cfg, "BIIE-AI/ace2_binding", "binary binding")

    out = capsys.readouterr().out
    assert captured["run_id"] == "t-stub"
    assert captured["task"] == "binary binding"  # auto mode passes the raw task
    assert captured["restart"] is False
    assert "step 0" in out  # log streamer rendered orchestrator output
    assert "completed" in out


def test_interactive_launch_passes_enriched_task(tmp_path, capsys, monkeypatch):
    import rich.prompt as rp

    import capo.cli.app as app
    import capo.cli.chat as chat
    from capo.cli.config import load_config

    cfg = load_config(_write_cfg(tmp_path, cli_mode="interactive", run_id="t-int",
                                 output_dir=str(tmp_path / "t-int")))
    # a realistic plan, built through the REAL enrichment path
    plan = chat._finalize(
        cfg, cfg.runs_root,
        {"objective": "binary binding classification", "dataset_ref": "BIIE-AI/ace2_binding",
         "fine_tune_strategy": "lora", "max_cost_usd": 20, "mode": "fine-tune"},
        "binary binding classification",
    )

    captured: dict = {}
    monkeypatch.setattr(chat, "run_chat",
                        lambda c, r, initial_intent=None, show_welcome=True: plan)
    monkeypatch.setattr(rp.Confirm, "ask", staticmethod(lambda *a, **k: True))
    monkeypatch.setattr(app, "_build_orchestrator", lambda run: _FakeOrch(captured))
    monkeypatch.setattr(app, "_assert_api_keys", lambda: None)
    monkeypatch.setattr(app, "run_console", lambda run_id, run_dir, stop, runs_root, abort_event=None: stop.wait())

    app._interactive_launch(cfg)

    task = captured["task"]
    # interactive mode feeds the orchestrator the structured task brief (→ task.md)
    assert task.startswith("# Task:") and "## Objective" in task
    assert "Data modality: protein sequences" in task
    assert "Fine-tune strategy: lora" in task
    assert "Budget ceiling: $20" in task
    assert captured["restart"] is False
    assert "Ready to launch" in capsys.readouterr().out


def test_interactive_launch_quit_is_noop(tmp_path, monkeypatch):
    import capo.cli.app as app
    import capo.cli.chat as chat
    from capo.cli.config import load_config

    cfg = load_config(_auto_cfg(tmp_path))
    called = {"orch": False}
    monkeypatch.setattr(chat, "run_chat",
                        lambda c, r, initial_intent=None, show_welcome=True: None)  # user quit
    monkeypatch.setattr(app, "_assert_api_keys", lambda: None)
    monkeypatch.setattr(app, "_build_orchestrator",
                        lambda run: called.__setitem__("orch", True))
    app._interactive_launch(cfg)  # must return cleanly without launching
    assert called["orch"] is False


def test_auto_launch_missing_task_errors(tmp_path):
    import typer

    import capo.cli.app as app
    from capo.cli.config import load_config

    cfg = load_config(_write_cfg(tmp_path, cli_mode="auto", task=None, task_file=None))
    with pytest.raises(typer.Exit) as exc:
        app._auto_launch(cfg, None, None)
    assert exc.value.exit_code == 1


class _PausedOrch:
    """Stub whose run returns a run paused for cost-overrun confirmation."""

    def __init__(self, captured):
        self._c = captured

    def run_sync(self, task_description, run_id, output_dir, restart_from_checkpoint=False):
        od = Path(output_dir)
        (od / "outputs").mkdir(parents=True, exist_ok=True)
        (od / "outputs" / "run.log").write_text("20:00:00 [fine-tuning] gate paused\n")
        reports = od / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        (reports / "pending_question.json").write_text(json.dumps({
            "header": "Accept overrun?",
            "question": "Projected cost $30.00 exceeds your budget $20.00. Accept and launch?",
            "options": [
                {"label": "accept", "description": "Launch at projected $30.00"},
                {"label": "reject", "description": "Replace with next candidate or abort"},
            ],
            "answer_target": "cost.accept_overrun",
        }))
        (od / "state.json").write_text(
            json.dumps({"current_phase": "pre_launch", "paused": True,
                        "pause_reason": "cost_accept_overrun",
                        "pending_question_path": "reports/pending_question.json"})
        )

        class _R:
            state = "unknown"
            finetuned_model_path = None
            trackio_url = None

        return _R()


def test_auto_launch_surfaces_pause_for_budget_confirmation(tmp_path, capsys, monkeypatch):
    import capo.cli.app as app
    from capo.cli.config import load_config

    cfg = load_config(_auto_cfg(tmp_path))
    monkeypatch.setattr(app, "_build_orchestrator", lambda run: _PausedOrch({}))
    monkeypatch.setattr(app, "_assert_api_keys", lambda: None)
    monkeypatch.setattr(app, "run_console", lambda run_id, run_dir, stop, runs_root, abort_event=None: stop.wait())
    # auto mode is non-interactive: it must surface the question but never prompt/resume.
    resumed = {"called": False}
    monkeypatch.setattr("capo.persistence.resume.resume_run",
                        lambda *a, **k: resumed.update(called=True))

    app._auto_launch(cfg, "BIIE-AI/ace2_binding", "binary binding")

    out = capsys.readouterr().out
    assert "paused" in out and "cost_accept_overrun" in out
    assert "capo resume t-stub" in out  # points the user at the resume path
    assert "Action needed" in out and "Accept overrun?" in out  # the question is surfaced
    assert resumed["called"] is False  # auto mode never auto-resumes


def test_app_resume_delegates_to_resume_run(monkeypatch):
    import typer

    import capo.cli.app as app
    import capo.persistence.resume as resume_mod

    seen = {}
    monkeypatch.setattr(resume_mod, "resume_run", lambda run_id, answer=None: seen.update(
        run_id=run_id, answer=answer) or 0)
    with pytest.raises(typer.Exit) as exc:
        app.resume("some-run", answer="accept")
    assert exc.value.exit_code == 0
    assert seen == {"run_id": "some-run", "answer": "accept"}


# === gradient logo ===─


def test_gradient_hex_endpoints_and_clamp():
    from capo.cli.colors import LOGO_C0, LOGO_C1, LOGO_C2, gradient_hex

    assert gradient_hex(0.0).upper() == LOGO_C0.upper()
    assert gradient_hex(0.5).upper() == LOGO_C1.upper()
    assert gradient_hex(1.0).upper() == LOGO_C2.upper()
    assert gradient_hex(-1).upper() == LOGO_C0.upper()  # clamps
    assert gradient_hex(2).upper() == LOGO_C2.upper()


# === log streamer (source chips, noise, continuation folding) ===


def test_log_source_chip_drops_inner_tag():
    from capo.cli.log_streamer import _parse_line, render_line

    p = _parse_line("16:16:43 [infra-agent] [lambda] Running preflight checks")
    assert p.chip == "infra"
    txt = render_line("16:16:43 [infra-agent] [lambda] Running preflight checks").plain
    assert "infra" in txt and "Running preflight checks" in txt
    assert "[lambda]" not in txt  # redundant inner tag dropped


def test_log_orchestrator_no_chip_and_noise():
    from capo.cli.log_streamer import _parse_line

    orch = _parse_line("16:14:39 [fine-tuning] gpu_preference resolved to: 1x A100")
    assert orch.chip == ""  # orchestrator stays white, no chip
    assert _parse_line("16:16:34 [cache] read=542835 creation=62018").msg_style == "log.noise"
    assert _parse_line("16:17:10 [infra-agent] [warning] retrying").msg_style == "log.warn"


def test_log_folds_continuation_lines():
    from capo.cli.log_streamer import _LogRenderer

    r = _LogRenderer()
    assert r.feed("16:16:51 [infra-agent] [status] Proceeding to Path C.")  # parent
    assert r.feed("STEP 2C — Check capacity")  # no ts → folded continuation, not dropped


def test_reasoning_traces_render_light_grey_italic():
    # #9: agent reasoning ([status] + synonyms) renders subtle grey italic, with
    # the tag dropped, while action lines keep their colour — and one set
    # (_REASONING_TAGS) is the single source of truth for that decision.
    from capo.cli.log_streamer import _msg_style, _REASONING_TAGS, render_line

    assert "status" in _REASONING_TAGS
    for tag in _REASONING_TAGS:
        assert _msg_style(tag, "x") == ("log.cont", False)  # all map to grey-italic

    line = render_line("10:00:01 [status] Checking whether the dataset has a split.")
    assert "[status]" not in line.plain                      # tag dropped → clean
    assert any(str(s.style) == "log.cont" for s in line.spans)  # message is grey-italic
    # an action/progress line is NOT styled as reasoning (kept visually distinct).
    prog = render_line("10:00:02 [progress] step 5/100 loss=0.41")
    assert all(str(s.style) != "log.cont" for s in prog.spans if s.style)


def test_reasoning_trace_truecolor_is_italic_grey():
    # the live pane renders each line to truecolor ANSI; reasoning must come out
    # italic (SGR 3) in NOISE grey (#AAAAAA = 170,170,170).
    from rich.console import Console

    from capo.cli.colors import THEME
    from capo.cli.log_streamer import render_line

    rec = Console(theme=THEME, highlight=False, force_terminal=True,
                  color_system="truecolor", width=100, record=True)
    rec.print(render_line("10:00:01 [reasoning] Comparing selected model against budget."))
    ansi = rec.export_text(styles=True)
    assert "3;38;2;170;170;170m" in ansi  # italic + grey on the message segment


# === config: generalized YAML writer + editor field coercion ===─


def test_set_yaml_value_types_and_comments(tmp_path):
    from capo.cli.config import set_yaml_value

    p = tmp_path / "c.yaml"
    p.write_text("# h\nmax_cost_usd: 50.0\nenable_memory: true\nmodel_id: a/b\n# tail\n")
    set_yaml_value(p, "max_cost_usd", 12.5)
    set_yaml_value(p, "enable_memory", False)
    set_yaml_value(p, "gpu_preference", "1x A100")  # absent → appended
    d = yaml.safe_load(p.read_text())
    assert d["max_cost_usd"] == 12.5 and d["enable_memory"] is False
    assert d["gpu_preference"] == "1x A100"
    assert p.read_text().count("#") == 2  # comments preserved


def test_config_edit_field_coercion(monkeypatch):
    import capo.cli.config as cfgmod

    monkeypatch.setattr("capo.cli.widgets.text_input", lambda label, default=None: "7")
    assert cfgmod._edit_field("Probe retries", "int", None, 3) == 7
    monkeypatch.setattr("capo.cli.widgets.text_input", lambda label, default=None: "notanum")
    assert cfgmod._edit_field("Max cost", "float", None, 1.0) is None  # bad → keep
    monkeypatch.setattr("capo.cli.widgets.select_one",
                        lambda label, choices, default=None, allow_other=True: "no")
    assert cfgmod._edit_field("HF research", "bool", None, True) is False


# === widgets ============─


def test_widgets_select_one(monkeypatch):
    import capo.cli.widgets as w

    monkeypatch.setattr(w, "_select", lambda lines_fn, count, default_idx=0: 1)
    assert w.select_one("Strategy", ["linear-probe", "lora", "full"], allow_other=False) == "lora"
    # picking the appended "Other…" row falls through to free text
    monkeypatch.setattr(w, "_select", lambda lines_fn, count, default_idx=0: count - 1)
    monkeypatch.setattr(w, "text_input", lambda label, default=None: "custom")
    assert w.select_one("Strategy", ["linear-probe", "lora"]) == "custom"


# === chat layer =========


def test_chat_extract_json():
    from capo.cli.chat import _extract_json

    assert _extract_json('x {"reply":"hi","ready":true} y')["ready"] is True
    assert _extract_json('```json\n{"reply":"x",}\n```')["reply"] == "x"  # fenced + trailing comma
    assert _extract_json("not json") is None


def test_chat_finalize_fills_defaults(tmp_path):
    from capo.cli.chat import _finalize
    from capo.cli.config import load_config

    cfg = load_config(_write_cfg(tmp_path, output_dir=str(tmp_path / "r")))
    plan = _finalize(cfg, cfg.runs_root,
                     {"objective": "binding", "fine_tune_strategy": None}, "binding")
    assert plan.dataset_ref == cfg.dataset_ref  # null → default
    assert plan.fine_tune_strategy == cfg.fine_tune_strategy
    # the structured brief carries the budget ceiling and standard sections
    assert "Budget ceiling" in plan.enriched_description
    assert "## Deliverables" in plan.enriched_description


def test_chat_run_ready_flow(tmp_path, monkeypatch):
    import capo.cli.chat as chat
    from capo.cli.config import load_config

    cfg = load_config(_write_cfg(tmp_path, output_dir=str(tmp_path / "r")))
    monkeypatch.setattr(chat, "_make_runner", lambda cfg: object())
    monkeypatch.setattr(chat, "_make_session", lambda: object())
    monkeypatch.setattr(chat, "_call_model", lambda runner, prompt, max_turns=2: json.dumps(
        {"reply": "go", "ready": True,
         "task": {"objective": "x", "dataset_ref": "o/d", "fine_tune_strategy": "lora",
                  "max_cost_usd": 15}}))
    plan = chat.run_chat(cfg, cfg.runs_root, initial_intent="train o/d lora $15")
    assert plan.dataset_ref == "o/d" and plan.fine_tune_strategy == "lora"
    assert plan.max_cost_usd == 15.0


def test_chat_run_question_then_ready(tmp_path, monkeypatch):
    import capo.cli.chat as chat
    import capo.cli.widgets as widgets
    from capo.cli.config import load_config

    cfg = load_config(_write_cfg(tmp_path, output_dir=str(tmp_path / "r")))
    turns = iter([
        {"reply": "which?", "ready": False, "questions": [
            {"key": "fine_tune_strategy", "prompt": "Strategy?", "choices": ["lora", "full"]}]},
        {"reply": "go", "ready": True, "task": {"objective": "binding", "dataset_ref": "o/d"}},
    ])
    monkeypatch.setattr(chat, "_make_runner", lambda cfg: object())
    monkeypatch.setattr(chat, "_make_session", lambda: object())
    monkeypatch.setattr(chat, "_call_model",
                        lambda runner, prompt, max_turns=2: json.dumps(next(turns)))
    monkeypatch.setattr(widgets, "select_one", lambda *a, **k: "lora")
    plan = chat.run_chat(cfg, cfg.runs_root, initial_intent="train o/d")
    assert plan.fine_tune_strategy == "lora"  # answer populated the task field directly


def test_chat_quit_returns_none(tmp_path, monkeypatch):
    import capo.cli.chat as chat
    from capo.cli.config import load_config

    cfg = load_config(_write_cfg(tmp_path, output_dir=str(tmp_path / "r")))
    monkeypatch.setattr(chat, "_make_runner", lambda cfg: object())
    monkeypatch.setattr(chat, "_make_session", lambda: object())
    assert chat.run_chat(cfg, cfg.runs_root, initial_intent="/quit") is None


def test_salvage_reply():
    from capo.cli.chat import _salvage_reply

    assert _salvage_reply("Let's set up ESM2 on ACE2.") == "Let's set up ESM2 on ACE2."
    assert _salvage_reply("```\nhello there\n```") == "hello there"  # fences stripped
    assert _salvage_reply('{"reply": "x"') == ""  # half-formed JSON is never echoed
    assert _salvage_reply("   ") == ""


def test_chat_retries_then_launches(tmp_path, monkeypatch):
    """A clear request whose first model reply has no JSON must still launch
    (retry recovers the structured reply) — not be told 'I didn't follow'."""
    import capo.cli.chat as chat
    from capo.cli.config import load_config

    cfg = load_config(_write_cfg(tmp_path, output_dir=str(tmp_path / "r")))
    turns = iter([
        "Sure! Let me set that up for you.",  # first reply: prose, no JSON
        json.dumps({"reply": "launching", "ready": True,
                    "task": {"objective": "ace2 binding", "dataset_ref": "o/d",
                             "fine_tune_strategy": "lora", "max_cost_usd": 15}}),
    ])
    monkeypatch.setattr(chat, "_make_runner", lambda cfg: object())
    monkeypatch.setattr(chat, "_make_session", lambda: object())
    monkeypatch.setattr(chat, "_call_model", lambda runner, prompt, max_turns=2: next(turns))
    plan = chat.run_chat(cfg, cfg.runs_root, initial_intent="fine-tune esm2 on o/d lora $15")
    assert plan is not None and plan.dataset_ref == "o/d"  # retry recovered → launched


def test_chat_salvages_prose_instead_of_apology(tmp_path, monkeypatch, capsys):
    """When even the retry yields prose, show it as a reply (friendly) rather
    than the over-eager 'didn't follow' stonewall."""
    import capo.cli.chat as chat
    from capo.cli.config import load_config

    cfg = load_config(_write_cfg(tmp_path, output_dir=str(tmp_path / "r")))
    monkeypatch.setattr(chat, "_make_runner", lambda cfg: object())
    monkeypatch.setattr(chat, "_make_session", lambda: object())
    monkeypatch.setattr(chat, "_call_model",
                        lambda runner, prompt, max_turns=2: "I can help — which dataset?")
    monkeypatch.setattr(chat, "_next_message", lambda *a, **k: None)  # user quits next
    plan = chat.run_chat(cfg, cfg.runs_root, initial_intent="help me train something")
    assert plan is None
    out = capsys.readouterr().out
    assert "I can help — which dataset?" in out  # prose surfaced as a reply
    assert "didn't" not in out  # never the over-eager apology


# === #5 post-pipeline interactive chat ========================================


def _finished_run_dir(root: Path, run_id: str = "r1") -> Path:
    """A minimal finished-run dir: final_summary + a couple of artifacts."""
    rd = root / run_id
    (rd / "reports").mkdir(parents=True)
    (rd / "outputs").mkdir(parents=True)
    (rd / "reports" / "final_summary.json").write_text(json.dumps({
        "terminal_state": "completed",
        "final_metrics": {"val_mcc": 0.92, "val_auroc": 0.99},
        "final_model_path": "/remote/checkpoints/best",
        "actual_cost_usd": 1.34,
    }))
    (rd / "task.md").write_text("# Task")
    (rd / "outputs" / "metrics.jsonl").write_text('{"step":1}\n')
    return rd


def test_post_run_context_summarizes_run(tmp_path):
    import capo.cli.chat as chat

    rd = _finished_run_dir(tmp_path)
    ctx = chat._post_run_context("r1", rd, result=None)
    assert "finished run: r1" in ctx
    assert "terminal_state: completed" in ctx
    assert "val_mcc=0.92" in ctx                       # metrics surfaced
    assert "final_model_path" in ctx
    # artifacts present are named so the agent knows what it can read
    assert "task.md" in ctx and "outputs/metrics.jsonl" in ctx
    assert "reports/final_summary.json" in ctx


def test_post_run_chat_qa_then_quit(tmp_path, monkeypatch):
    import capo.cli.chat as chat
    from capo.cli.config import load_config

    cfg = load_config(_write_cfg(tmp_path, output_dir=str(tmp_path / "r")))
    rd = _finished_run_dir(tmp_path)
    inputs = iter(["explain the results", "/quit"])
    monkeypatch.setattr(chat, "_make_runner", lambda *a, **k: object())
    monkeypatch.setattr(chat, "_make_session", lambda: object())
    monkeypatch.setattr(chat, "_read_user", lambda session: next(inputs))
    monkeypatch.setattr(chat, "_call_model", lambda runner, prompt, max_turns=2: json.dumps(
        {"reply": "Your validation MCC was 0.92 — a strong result.", "ready": False}))

    plan = chat.run_post_run_chat(cfg, cfg.runs_root, "r1", rd)
    assert plan is None  # a Q&A turn then a quit → no launch


def test_post_run_chat_routes_finetuning_to_plan(tmp_path, monkeypatch):
    import capo.cli.chat as chat
    from capo.cli.config import load_config

    cfg = load_config(_write_cfg(tmp_path, output_dir=str(tmp_path / "r")))
    rd = _finished_run_dir(tmp_path)
    captured: dict = {}

    def _runner(*a, **k):  # the post-run runner gets file tools + the run dir cwd
        captured["mk"] = k
        return object()

    monkeypatch.setattr(chat, "_make_runner", _runner)
    monkeypatch.setattr(chat, "_make_session", lambda: object())
    monkeypatch.setattr(chat, "_read_user", lambda session: "fine-tune again but with lora")
    monkeypatch.setattr(chat, "_call_model", lambda runner, prompt, max_turns=2: json.dumps(
        {"reply": "Launching another run with LoRA.", "ready": True,
         "task": {"objective": "ace2 binding", "dataset_ref": "BIIE-AI/ace2_binding",
                  "fine_tune_strategy": "lora", "max_cost_usd": 20, "mode": "fine-tune"}}))

    plan = chat.run_post_run_chat(cfg, cfg.runs_root, "r1", rd)
    assert plan is not None                                   # routed to a launch plan
    assert plan.fine_tune_strategy == "lora" and plan.dataset_ref == "BIIE-AI/ace2_binding"
    assert plan.enriched_description.startswith("# Task:")    # same task.md enrichment path
    # the post-run runner is built with read-only file tools scoped to the run dir
    assert captured["mk"]["allowed_tools"] == ["Read", "Grep", "Glob"]
    assert captured["mk"]["cwd"] == str(rd)
    assert captured["mk"]["prompt_name"] == "cli/post_run_chat"


def test_post_run_interaction_launches_through_same_pipeline(tmp_path, monkeypatch):
    import capo.cli.app as app
    import capo.cli.chat as chat
    from capo.cli.config import load_config

    cfg = load_config(_write_cfg(tmp_path, output_dir=str(tmp_path / "r2")))
    plan = chat._finalize(
        cfg, cfg.runs_root,
        {"objective": "thermostability regression", "dataset_ref": "o/therm",
         "fine_tune_strategy": "full", "max_cost_usd": 30, "mode": "fine-tune"},
        "thermostability regression",
    )
    launched: dict = {}
    # the chat returns a launch plan once, then (on the relaunched run's own chat) None
    plans = iter([plan, None])
    # _post_run_interaction does `from .chat import run_post_run_chat` at call time.
    monkeypatch.setattr(chat, "run_post_run_chat", lambda *a, **k: next(plans, None))
    monkeypatch.setattr(app, "_confirm_launch", lambda p: True)
    monkeypatch.setattr(app, "_build_orchestrator", lambda run: object())

    def _fake_launch(orch, **kwargs):
        launched.update(kwargs)

    monkeypatch.setattr(app, "_launch", _fake_launch)

    app._post_run_interaction(cfg, run_id="r1", run_dir=tmp_path / "r1",
                              runs_root=cfg.runs_root, result=None)
    # the plan is launched through the SAME pipeline: fresh run, interactive, cfg threaded
    assert launched["from_start"] is True and launched["interactive"] is True
    assert launched["cfg"] is cfg
    assert launched["task_description"].startswith("# Task:")  # enriched task.md brief


# === #10 abort: sync remote data, then optionally terminate the instance ======


def _remote_run_dir(root: Path, run_id: str = "r1", *, with_instance: bool = True) -> Path:
    """A run dir that looks like one which provisioned an instance (infra.json +
    session state), so _abort_cleanup has remote handles to act on."""
    rd = root / run_id
    rd.mkdir(parents=True)
    if with_instance:
        (rd / "infra.json").write_text(
            json.dumps({"instance_id": "i-123", "ssh_alias": "lambda-i-123"})
        )
    (rd / "state.json").write_text(json.dumps({
        "run_id": run_id, "key_path": "/home/u/.ssh/k", "ssh_key_name": "mykey",
        "remote_run_dir": f"~/capo_runs/{run_id}",
    }))
    return rd


def test_read_remote_run_info_reads_infra_and_state(tmp_path):
    import capo.cli.app as app

    info = app._read_remote_run_info("r1", _remote_run_dir(tmp_path))
    assert info is not None
    ssh_alias, key_path, remote_run_dir, instance_id, ssh_key_names = info
    assert ssh_alias == "lambda-i-123"
    assert key_path == "/home/u/.ssh/k"
    assert remote_run_dir == "~/capo_runs/r1"
    assert instance_id == "i-123"
    assert ssh_key_names == ["mykey"]


def test_read_remote_run_info_none_without_infra(tmp_path):
    import capo.cli.app as app

    rd = tmp_path / "r1"
    rd.mkdir()
    assert app._read_remote_run_info("r1", rd) is None  # never reached an instance


def _stub_remote(monkeypatch, *, instance_down: bool = True) -> dict:
    """Stub the remote ops _abort_cleanup uses; record their calls. By default the
    post-terminate verification sees the instance as gone (get_instance 404s)."""
    import types

    import capo.remote.lambda_session as ls
    import capo.remote.run_manager as rm
    import capo.remote.rsync_manager as rsm

    calls: dict = {}
    monkeypatch.setattr(rm, "stop_remote_run", lambda *a, **k: calls.__setitem__("stop", (a, k)))
    monkeypatch.setattr(rsm, "download_run_outputs", lambda *a, **k: calls.__setitem__("pull", (a, k)))
    monkeypatch.setattr(ls, "safe_terminate_instance",
                        lambda *a, **k: calls.__setitem__("term", (a, k)))

    def _get_instance(instance_id, api_key=None):
        calls.setdefault("verify", []).append(instance_id)
        if instance_down:
            raise RuntimeError("404 not found")            # gone → verified down
        return types.SimpleNamespace(status="active")      # still running
    monkeypatch.setattr(ls, "get_instance", _get_instance)
    return calls


def test_abort_cleanup_terminates_and_verifies(tmp_path, monkeypatch):
    import capo.cli.app as app

    rd = _remote_run_dir(tmp_path)
    calls = _stub_remote(monkeypatch)

    app._abort_cleanup("r1", rd, tmp_path, interactive=True)

    # stop → sync → terminate → verify, with NO yes/no prompt: an abort always tears
    # the instance down so a GPU is never left billing after the user stops a run.
    assert "stop" in calls and "pull" in calls and "term" in calls
    assert calls["term"][0][0] == "i-123"                  # right instance id
    assert calls["term"][0][1] == ["mykey"]                # ownership-checked terminate
    pulled = calls["pull"][1]["subpaths"]
    assert "checkpoints/" in pulled                        # pulled checkpoints…
    assert "results/" in pulled                            # …AND eval results before teardown
    assert "outputs/" in pulled
    assert calls.get("verify") == ["i-123"]                # verified the instance is down


def test_abort_cleanup_handles_terminate_failure(tmp_path, monkeypatch):
    import capo.remote.lambda_session as ls

    import capo.cli.app as app

    rd = _remote_run_dir(tmp_path)
    calls = _stub_remote(monkeypatch)
    monkeypatch.setattr(ls, "safe_terminate_instance",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("api down")))

    app._abort_cleanup("r1", rd, tmp_path, interactive=True)  # must not raise

    assert "stop" in calls and "pull" in calls   # data still synced off the box
    assert "verify" not in calls                  # gave up after the terminate error


def test_abort_cleanup_noop_without_instance(tmp_path, monkeypatch):
    import capo.cli.app as app

    rd = tmp_path / "r1"
    rd.mkdir()  # no infra.json → no instance was ever attached
    calls = _stub_remote(monkeypatch)
    app._abort_cleanup("r1", rd, tmp_path, interactive=True)
    assert calls == {}  # nothing remote to stop / sync / terminate


def test_verify_instance_down_gone(monkeypatch):
    import capo.remote.lambda_session as ls

    import capo.cli.app as app

    monkeypatch.setattr(ls, "get_instance",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("404")))
    assert app._verify_instance_down("i-1") is True   # 404 → gone


def test_verify_instance_down_terminating(monkeypatch):
    import types

    import capo.remote.lambda_session as ls

    import capo.cli.app as app

    monkeypatch.setattr(ls, "get_instance",
                        lambda *a, **k: types.SimpleNamespace(status="terminating"))
    assert app._verify_instance_down("i-1") is True   # winding down counts as down


def test_verify_instance_down_times_out(monkeypatch):
    import types

    import capo.remote.lambda_session as ls

    import capo.cli.app as app

    monkeypatch.setattr(ls, "get_instance",
                        lambda *a, **k: types.SimpleNamespace(status="active"))
    # timeout_s=0 → returns False on the first check without ever sleeping
    assert app._verify_instance_down("i-1", timeout_s=0.0) is False


def test_run_and_report_runs_abort_cleanup_on_interrupt(tmp_path, monkeypatch):
    import capo.cli.app as app

    rd = tmp_path / "r1"
    (rd / "outputs").mkdir(parents=True)
    (rd / "state.json").write_text(json.dumps({"current_phase": "training"}))

    class _Orch:
        def run_sync(self, **kwargs):
            raise KeyboardInterrupt  # /abort or Ctrl+C reaches run_sync as SIGINT

    recorded: dict = {}
    monkeypatch.setattr(app, "run_console", lambda *a, **k: None)  # don't open a real prompt
    monkeypatch.setattr(
        app, "_abort_cleanup",
        lambda run_id, run_dir, runs_root, *, interactive: recorded.update(
            run_id=run_id, interactive=interactive),
    )
    app._run_and_report(_Orch(), {}, run_id="r1", run_dir=rd, runs_root=tmp_path,
                        interactive=True, from_start=True, cfg=None)
    assert recorded == {"run_id": "r1", "interactive": True}  # abort path → cleanup ran


def test_run_and_report_detects_abort_via_event(tmp_path, monkeypatch):
    """Even if the orchestrator swallows the SIGINT and run_sync returns normally, a
    set abort_event must route to the abort cleanup — never the post-run chat (the
    bug that jumped an abort straight back to 'ready to launch')."""
    import threading

    import capo.cli.app as app

    rd = tmp_path / "r1"
    (rd / "outputs").mkdir(parents=True)
    (rd / "state.json").write_text(json.dumps({"current_phase": "training"}))

    ready = threading.Event()

    def fake_run_console(run_id, run_dir, stop, runs_root, abort_event=None):
        abort_event.set()   # the /abort path sets this from inside the console
        ready.set()

    class _Orch:
        def run_sync(self, **kwargs):
            ready.wait(timeout=2)                  # console sets the flag before we return
            return {"terminal_state": "completed"}  # normal return, no KeyboardInterrupt

    recorded: dict = {}
    monkeypatch.setattr(app, "run_console", fake_run_console)
    monkeypatch.setattr(app, "_abort_cleanup", lambda *a, **k: recorded.__setitem__("abort", True))
    monkeypatch.setattr(app, "_post_run_interaction",
                        lambda *a, **k: recorded.__setitem__("post_run", True))

    app._run_and_report(_Orch(), {}, run_id="r1", run_dir=rd, runs_root=tmp_path,
                        interactive=True, from_start=True, cfg=object())

    assert recorded.get("abort") is True   # abort path taken despite the clean return
    assert "post_run" not in recorded      # NOT the post-run chat / ready-to-launch


def test_run_and_report_raw_interrupt_declined_keeps_run(tmp_path, monkeypatch, capsys):
    """A raw Ctrl+C (one that bypassed the run view's inline confirm) is re-confirmed
    in the main thread. Declining leaves everything as-is — no teardown, no front
    door — so a stray interrupt never destroys the run + GPU."""
    import capo.cli.app as app

    rd = tmp_path / "r1"
    (rd / "outputs").mkdir(parents=True)
    (rd / "state.json").write_text(json.dumps({"current_phase": "training"}))

    class _Orch:
        def run_sync(self, **kwargs):
            raise KeyboardInterrupt  # no abort_flag → an unconfirmed raw interrupt

    recorded: dict = {}
    monkeypatch.setattr(app, "run_console", lambda *a, **k: None)
    monkeypatch.setattr(app, "_confirm_abort_teardown", lambda: False)  # user declines
    monkeypatch.setattr(app, "_abort_cleanup", lambda *a, **k: recorded.__setitem__("abort", True))

    app._run_and_report(_Orch(), {}, run_id="r1", run_dir=rd, runs_root=tmp_path,
                        interactive=True, from_start=True, cfg=None)

    assert "abort" not in recorded                          # declined → nothing torn down
    assert "keeping the run" in capsys.readouterr().out.lower()


def test_confirm_abort_teardown_noninteractive_true(monkeypatch):
    """With no tty to ask (tests, pipes), an interrupt stands — return True so the
    teardown proceeds rather than hanging on a prompt no one can answer."""
    import capo.cli.app as app

    monkeypatch.setattr(app.sys.stdin, "isatty", lambda: False)
    assert app._confirm_abort_teardown() is True


# === #4 strengthen first orchestrator (config-gated, default off) =============


def test_agent_runner_effort_skills_default_off():
    from capo.orchestration.agent_runner import AgentRunner

    r = AgentRunner(model_name="claude-sonnet-4-6", allowed_tools=["Read"])
    assert r.effort is None and r.skills is None
    opts = r._build_options()
    # untouched → ClaudeAgentOptions keeps its own defaults (None), so a run is
    # byte-for-byte what it was before the knob existed.
    assert opts.effort is None and opts.skills is None


def test_agent_runner_effort_skills_when_set():
    from capo.orchestration.agent_runner import AgentRunner

    r = AgentRunner(model_name="claude-sonnet-4-6", allowed_tools=["Read"],
                    effort="high", skills="all")
    opts = r._build_options()
    assert opts.effort == "high" and opts.skills == "all"


def test_agent_runner_skills_normalization():
    from capo.orchestration.agent_runner import AgentRunner

    norm = AgentRunner._normalize_skills
    assert norm(None) is None and norm("") is None and norm("   ") is None
    assert norm("all") == "all" and norm("ALL") == "all"
    assert norm("esm, clustering ,uniprot") == ["esm", "clustering", "uniprot"]
    assert norm(["a", "b"]) == ["a", "b"] and norm([]) is None


def test_config_loads_orchestrator_knobs(tmp_path):
    from capo.cli.config import load_config

    cfg = load_config(_write_cfg(tmp_path))  # keys absent → off
    assert cfg.orchestrator_effort is None and cfg.orchestrator_skills is None

    cfg2 = load_config(_write_cfg(tmp_path, orchestrator_effort="high",
                                  orchestrator_skills="all"))
    assert cfg2.orchestrator_effort == "high" and cfg2.orchestrator_skills == "all"


def test_orchestrator_knobs_reach_only_main_runner(tmp_path):
    """The gated effort/skills knobs land on the FIRST/main orchestrator only.
    The pre-launch sub-runners + finalizer are ISOLATED from the config knob —
    their own (deliberately hardcoded) effort is independent of it, so it must be
    identical whether the knob is on or off — and the skills knob never reaches
    them at all. Default config leaves the main runner off."""
    import capo.cli.app as app
    from capo.cli.config import load_config
    from capo.cli.questionnaire import RunConfig

    def _build(**knobs):
        cfg = load_config(_write_cfg(tmp_path, enable_hf_research=False,
                                     enable_memory=False, **knobs))
        return app._build_orchestrator(RunConfig.from_config(cfg))

    on = _build(orchestrator_effort="high", orchestrator_skills="all")
    off = _build()  # keys absent → knob off

    # the knob reaches the MAIN orchestrator ...
    assert on._orchestrator.effort == "high" and on._orchestrator.skills == "all"
    assert off._orchestrator.effort is None and off._orchestrator.skills is None

    # ... and does NOT leak past it: each sub-runner's effort/skills is identical
    # with the knob on vs off (the gated config never crosses into a sub-runner).
    for name in ("_infra_runner", "_data_runner", "_model_runner", "_finalizer_runner"):
        assert getattr(on, name).effort == getattr(off, name).effort, name
        assert getattr(on, name).skills == getattr(off, name).skills, name
    # the skills knob in particular never reaches a sub-runner.
    assert on._data_runner.skills is None and on._finalizer_runner.skills is None


# === clean API-key error =========


def test_api_key_error_block(capsys):
    import capo.cli.app as app

    app._print_key_error({"ANTHROPIC_API_KEY"})
    out = capsys.readouterr().out
    assert "needs your API keys" in out
    assert "Please set HF_TOKEN, ANTHROPIC_API_KEY and LAMBDA_API_KEY" in out
    assert "ANTHROPIC_API_KEY" in out and "LAMBDA_API_KEY" in out and "HF_TOKEN" in out
    assert ".env file" in out


# === new polish: SDK-noise, task brief, config Save/Cancel, landing, thinking ===


def test_sdk_logging_suppressed():
    # importing the progress emitter must mute the claude_agent_sdk INFO banner
    # at the source (it is on the import path of both the CLI and the script).
    import logging

    from capo.observability import progress  # noqa: F401  (import has the effect)

    lg = logging.getLogger("claude_agent_sdk._internal.transport.subprocess_cli")
    assert lg.getEffectiveLevel() >= logging.WARNING


def test_build_task_markdown_structure(tmp_path):
    from capo.cli.run_planner import build_task_markdown

    md, intent = build_task_markdown(
        objective="Predict binary ACE2 binding",
        mode="fine-tune",
        dataset_ref="BIIE-AI/ace2_binding",
        fine_tune_strategy="lora",
        max_cost_usd=20.0,
        gpu_preference="1x A100",
        model_id="facebook/esm2_t12_35M_UR50D",
        runs_root=tmp_path,
        organism="human",
        target="ACE2 binding",
    )
    for section in ("# Task:", "## Objective", "## Dataset", "## Training Strategy",
                    "## Evaluation", "## Deliverables", "## Constraints"):
        assert section in md
    assert "Organism / species: human" in md
    assert "Fine-tune strategy: lora" in md
    assert intent.inferred_modality == "protein_sequence"


def test_build_task_markdown_pretrain_mode(tmp_path):
    from capo.cli.run_planner import build_task_markdown

    md, _ = build_task_markdown(
        objective="pretrain a small protein LM", mode="pre-train",
        dataset_ref="o/seqs", fine_tune_strategy="full", max_cost_usd=10.0,
        gpu_preference=None, model_id=None, runs_root=tmp_path,
    )
    assert "pre-train from a custom architecture" in md
    assert "Fine-tune strategy" not in md  # no strategy line when pre-training


def test_config_editor_save_writes_pending(tmp_path, monkeypatch):
    import capo.cli.config as cfgmod
    import capo.cli.widgets as widgets

    p = _write_cfg(tmp_path, max_cost_usd=50.0)
    cfg = cfgmod.load_config(p)
    idx = next(i for i, (_l, a, _k, _c) in enumerate(cfgmod._FIELDS) if a == "max_cost_usd")
    seq = iter([idx, len(cfgmod._FIELDS)])  # edit max cost, then [ Save ]
    monkeypatch.setattr(widgets, "_select", lambda fn, n, default_idx=0: next(seq))
    monkeypatch.setattr(cfgmod, "_edit_field", lambda *a: 30.0)
    cfgmod.interactive_config_editor(cfg)
    assert "max_cost_usd: 30.0" in p.read_text()
    assert cfg.max_cost_usd == 30.0


def test_config_editor_cancel_discards(tmp_path, monkeypatch):
    import capo.cli.config as cfgmod
    import capo.cli.widgets as widgets

    p = _write_cfg(tmp_path, max_cost_usd=50.0)
    cfg = cfgmod.load_config(p)
    idx = next(i for i, (_l, a, _k, _c) in enumerate(cfgmod._FIELDS) if a == "max_cost_usd")
    seq = iter([idx, len(cfgmod._FIELDS) + 1])  # edit, then [ Cancel ]
    monkeypatch.setattr(widgets, "_select", lambda fn, n, default_idx=0: next(seq))
    monkeypatch.setattr(cfgmod, "_edit_field", lambda *a: 99.0)
    cfgmod.interactive_config_editor(cfg)
    assert "max_cost_usd: 50.0" in p.read_text()  # nothing written
    assert cfg.max_cost_usd == 50.0


def test_command_help_table_lists_commands(capsys):
    from capo.cli.chat import _command_help_table
    from capo.cli.colors import console

    console.print(_command_help_table())
    out = capsys.readouterr().out
    for cmd in ("/help", "/config", "/history", "/quit"):
        assert cmd in out


def test_thinking_indicator_noninteractive_is_safe():
    # under pytest stdout is not a TTY → no animation thread, no output, no crash
    from capo.cli.chat import _Thinking

    with _Thinking():
        pass


# === shared command registry + deterministic slash handling ===


def test_command_registry_full_help_and_quitwords(capsys):
    from capo.cli.commands import COMMANDS, command_help_table, is_quit
    from capo.cli.colors import console

    # /help lists *every* command, not just the landing subset
    console.print(command_help_table())
    out = capsys.readouterr().out
    for cmd in ("/help", "/config", "/history", "/health", "/abort",
                "/tune", "/prune-memory", "/quit"):
        assert cmd in out
    # scope filtering: /retune is chat-only, /health is run-only
    run_names = [n for n, _d, s in COMMANDS if s in ("run", "both")]
    assert "/health" in run_names and "/retune" not in run_names
    # plain quit/exit count as quit, a real message does not
    assert is_quit("quit") and is_quit("EXIT") and is_quit("/quit")
    assert not is_quit("train o/d")


def test_resolve_slash_prefix_and_ambiguity():
    from capo.cli.commands import resolve_slash

    names = ["/help", "/config", "/history", "/health", "/status", "/quit"]
    # exact name resolves to itself, even though it prefixes nothing here
    assert resolve_slash("/status", names) == ("/status", ["/status"])
    # a unique prefix expands to the one command it can mean
    assert resolve_slash("/heal", names) == ("/health", ["/health"])
    assert resolve_slash("/q", names) == ("/quit", ["/quit"])
    # case-insensitive on the token
    assert resolve_slash("/HEAL", names) == ("/health", ["/health"])
    # an ambiguous prefix resolves to nothing and reports the candidates
    resolved, matches = resolve_slash("/he", names)
    assert resolved is None and matches == ["/help", "/health"]
    # an unknown token resolves to nothing with no candidates
    assert resolve_slash("/zzz", names) == (None, [])


def test_chat_handle_command_is_deterministic(tmp_path):
    import capo.cli.chat as chat
    from capo.cli.config import load_config

    cfg = load_config(_write_cfg(tmp_path, output_dir=str(tmp_path / "r")))
    # quit words and /quit → quit (no model)
    assert chat._handle_command("/quit", cfg, cfg.runs_root)[0] == "quit"
    assert chat._handle_command("quit", cfg, cfg.runs_root)[0] == "quit"
    assert chat._handle_command("exit", cfg, cfg.runs_root)[0] == "quit"
    # a utility command is "handled" (runs, then keeps chatting — no model reply)
    assert chat._handle_command("/help", cfg, cfg.runs_root) == ("handled", None)
    # an unknown slash command is still handled deterministically (never the model)
    assert chat._handle_command("/bogus", cfg, cfg.runs_root) == ("handled", None)
    # a real message, or a /tune instruction, is routed to the model
    assert chat._handle_command("train o/d", cfg, cfg.runs_root) == ("model", "train o/d")
    assert chat._handle_command("/tune use lora", cfg, cfg.runs_root) == ("model", "use lora")
    # /tune with no argument is just usage text (handled, no model)
    assert chat._handle_command("/tune", cfg, cfg.runs_root) == ("handled", None)


def test_chat_handle_command_prefix_expansion(tmp_path, capsys):
    import capo.cli.chat as chat
    from capo.cli.config import load_config

    cfg = load_config(_write_cfg(tmp_path, output_dir=str(tmp_path / "r")))
    # a unique prefix expands to the full command: /q → /quit, /hi → /history
    assert chat._handle_command("/q", cfg, cfg.runs_root)[0] == "quit"
    assert chat._handle_command("/hi", cfg, cfg.runs_root) == ("handled", None)
    # a unique prefix carries its argument through: /ret use lora → /retune
    assert chat._handle_command("/ret use lora", cfg, cfg.runs_root) == ("model", "use lora")
    # an ambiguous prefix runs nothing; it lists candidates and asks to narrow
    assert chat._handle_command("/h", cfg, cfg.runs_root) == ("handled", None)
    out = capsys.readouterr().out
    assert "/help" in out and "/history" in out and "type more" in out


def test_chat_slash_command_does_not_call_model(tmp_path, monkeypatch):
    import capo.cli.chat as chat
    from capo.cli.config import load_config

    cfg = load_config(_write_cfg(tmp_path, output_dir=str(tmp_path / "r")))
    calls = {"n": 0}

    def _fake_call(runner, prompt, max_turns=2):
        calls["n"] += 1
        return json.dumps({"reply": "go", "ready": True,
                           "task": {"objective": "x", "dataset_ref": "o/d"}})

    monkeypatch.setattr(chat, "_make_runner", lambda c: object())
    monkeypatch.setattr(chat, "_make_session", lambda: object())
    monkeypatch.setattr(chat, "_call_model", _fake_call)
    # first a /help (must NOT hit the model), then a real message (one model call)
    inputs = iter(["/help", "train o/d"])
    monkeypatch.setattr(chat, "_read_user", lambda session: next(inputs))

    plan = chat.run_chat(cfg, cfg.runs_root)
    assert plan is not None and calls["n"] == 1  # exactly one model call, after the real message


# === /prune-memory operates on runs_index.md (no LLM) ===


def _write_index(tmp_path: Path) -> Path:
    idx = tmp_path / "runs_index.md"
    idx.write_text(
        "---\nrun_id: run-aaa\ntask_summary: ACE2 binding\n"
        "report_path: capo/run-aaa/RUN_REPORT.md\n---\n\n"
        "---\nrun_id: run-bbb\ntask_summary: thermostability\n"
        "report_path: capo/run-bbb/RUN_REPORT.md\n---\n"
    )
    return idx


def test_remove_index_blocks(tmp_path):
    from capo.memory.run_report import read_index_blocks, remove_index_blocks

    idx = _write_index(tmp_path)
    lock = tmp_path / ".lock"
    removed = remove_index_blocks(["run-aaa"], index_path=idx, lock_path=lock)
    assert removed == 1
    assert [b["run_id"] for b in read_index_blocks(idx)] == ["run-bbb"]
    # removing a non-existent id is a no-op
    assert remove_index_blocks(["nope"], index_path=idx, lock_path=lock) == 0


def test_prune_memory_direct_form(tmp_path, capsys):
    from capo.cli.memory import prune_memory
    from capo.memory.run_report import read_index_blocks

    idx = _write_index(tmp_path)
    prune_memory(idx, run_id="run-aaa")
    out = capsys.readouterr().out
    assert "Removed" in out and "run-aaa" in out
    assert [b["run_id"] for b in read_index_blocks(idx)] == ["run-bbb"]


def test_prune_memory_interactive_confirm(tmp_path, monkeypatch):
    import capo.cli.memory as mem
    import capo.cli.widgets as widgets
    from capo.memory.run_report import read_index_blocks

    idx = _write_index(tmp_path)
    # tick run index 0, then choose [ Confirm ] (index == n == 2)
    seq = iter([0, 2])
    monkeypatch.setattr(widgets, "_select", lambda fn, n, default_idx=0: next(seq))
    mem.prune_memory(idx)
    assert [b["run_id"] for b in read_index_blocks(idx)] == ["run-bbb"]


def test_prune_memory_interactive_cancel_keeps_all(tmp_path, monkeypatch):
    import capo.cli.memory as mem
    import capo.cli.widgets as widgets
    from capo.memory.run_report import read_index_blocks

    idx = _write_index(tmp_path)
    seq = iter([0, 3])  # tick run 0, then [ Cancel ] (index == n + 1 == 3)
    monkeypatch.setattr(widgets, "_select", lambda fn, n, default_idx=0: next(seq))
    mem.prune_memory(idx)
    assert [b["run_id"] for b in read_index_blocks(idx)] == ["run-aaa", "run-bbb"]  # untouched


# === run console: running-time clock + run-time /help ===


def test_elapsed_rprompt_keeps_running_time_only(monkeypatch):
    import capo.cli.run_console as rc
    from prompt_toolkit.formatted_text import to_formatted_text

    monkeypatch.setattr(rc.time, "monotonic", lambda: 75.0)
    txt = "".join(t for _s, t in to_formatted_text(rc.elapsed_rprompt(0.0)))
    assert "00:01:15" in txt  # the running time is kept on the right edge
    assert "●" not in txt and "LIVE" not in txt  # no follow-the-cursor live marker


def test_run_summary_box(capsys, tmp_path):
    from capo.cli.run_console import print_run_summary

    class _R:
        state = "completed"
        finetuned_model_path = "BIIE-AI/esm2-binding"
        trackio_url = "https://trackio/x"

    print_run_summary("run-xyz", tmp_path, result=_R())
    out = capsys.readouterr().out
    assert "completed" in out and "run-xyz" in out
    assert "BIIE-AI/esm2-binding" in out and "capo resume run-xyz" in out


def test_run_summary_box_pause(capsys, tmp_path):
    from capo.cli.run_console import print_run_summary

    class _R:
        state = "unknown"
        finetuned_model_path = None
        trackio_url = None

    print_run_summary(
        "t-stub", tmp_path, result=_R(), paused=True, pause_reason="cost_accept_overrun"
    )
    out = capsys.readouterr().out
    assert "paused" in out and "cost_accept_overrun" in out
    assert "capo resume t-stub" in out


# === pending-question surfacing (#3) ==========================================


def _cost_question() -> dict:
    return {
        "header": "Accept overrun?",
        "question": "Projected cost $30.00 exceeds your budget $20.00. Accept and launch?",
        "options": [
            {"label": "accept", "description": "Launch at projected $30.00"},
            {"label": "reject", "description": "Replace with next candidate or abort"},
        ],
        "answer_target": "cost.accept_overrun",
    }


def test_read_pending_question(tmp_path):
    from capo.cli.run_console import read_pending_question

    assert read_pending_question(tmp_path) is None  # nothing written yet

    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "pending_question.json").write_text(json.dumps(_cost_question()))
    q = read_pending_question(tmp_path)
    assert q is not None and q["answer_target"] == "cost.accept_overrun"

    # state.json's pending_question_path wins over the conventional location
    (tmp_path / "custom.json").write_text(json.dumps({"question": "hi", "answer_target": "x"}))
    q2 = read_pending_question(tmp_path, state={"pending_question_path": "custom.json"})
    assert q2["question"] == "hi"


def test_print_run_summary_renders_question(capsys, tmp_path):
    from capo.cli.run_console import print_run_summary

    class _R:
        state = "unknown"
        finetuned_model_path = None
        trackio_url = None

    print_run_summary(
        "t-stub", tmp_path, result=_R(), paused=True,
        pause_reason="cost_accept_overrun", pending_question=_cost_question(),
    )
    out = capsys.readouterr().out
    assert "Action needed" in out
    assert "Accept overrun?" in out  # header
    assert "exceeds your budget" in out  # question text
    assert "accept" in out and "reject" in out  # both option labels


def test_prompt_pending_answer_choice(monkeypatch):
    import rich.prompt as rp

    from capo.cli.run_console import prompt_pending_answer

    q = _cost_question()
    monkeypatch.setattr(rp.Prompt, "ask", staticmethod(lambda *a, **k: "1"))
    assert prompt_pending_answer(q) == "accept"  # 1-based index → first label
    monkeypatch.setattr(rp.Prompt, "ask", staticmethod(lambda *a, **k: "reject"))
    assert prompt_pending_answer(q) == "reject"  # label match (case-insensitive)
    monkeypatch.setattr(rp.Prompt, "ask", staticmethod(lambda *a, **k: ""))
    assert prompt_pending_answer(q) is None  # Enter → defer
    monkeypatch.setattr(rp.Prompt, "ask", staticmethod(lambda *a, **k: "later"))
    assert prompt_pending_answer(q) is None  # 'later' → defer


def test_prompt_pending_answer_free_text(monkeypatch):
    import rich.prompt as rp

    from capo.cli.run_console import prompt_pending_answer

    q = {
        "header": "Schema gap",
        "question": "Describe the label semantics.",
        "options": [{"label": "Provide details", "description": "Free-text answer"}],
        "answer_target": "profile.label_semantics",
    }
    monkeypatch.setattr(rp.Prompt, "ask", staticmethod(lambda *a, **k: "Kd in nM, lower binds"))
    assert prompt_pending_answer(q) == "Kd in nM, lower binds"  # verbatim free text
    monkeypatch.setattr(rp.Prompt, "ask", staticmethod(lambda *a, **k: "  "))
    assert prompt_pending_answer(q) is None  # blank → defer


def test_offer_inline_resume_reenters_live_view(monkeypatch, tmp_path):
    """An inline answer prepares the resume and re-enters the SAME run view
    (full TUI), instead of dropping to plain foreground."""
    import capo.cli.app as app

    monkeypatch.setattr(app, "prompt_pending_answer", lambda q: "accept")
    fake_orch, fake_kwargs = object(), {"resume_from_pause": True, "run_id": "t-stub"}
    monkeypatch.setattr("capo.persistence.resume.prepare_pause_resume",
                        lambda run_id, answer: (fake_orch, fake_kwargs))
    seen: dict = {}
    monkeypatch.setattr(app, "_run_and_report",
                        lambda orch, kw, **k: seen.update(orch=orch, kw=kw, k=k))
    app._offer_inline_resume(_cost_question(), run_id="t-stub", run_dir=tmp_path,
                             runs_root=tmp_path, interactive=True)
    assert seen["orch"] is fake_orch and seen["kw"] is fake_kwargs
    assert seen["k"]["run_id"] == "t-stub" and seen["k"]["from_start"] is False


def test_offer_inline_resume_falls_back_when_unpreparable(monkeypatch, tmp_path):
    """If a TUI re-entry can't be prepared, fall back to plain resume_run so the
    answer still lands."""
    import capo.cli.app as app

    monkeypatch.setattr(app, "prompt_pending_answer", lambda q: "accept")
    monkeypatch.setattr("capo.persistence.resume.prepare_pause_resume",
                        lambda run_id, answer: None)  # can't prepare
    called: dict = {}
    monkeypatch.setattr("capo.persistence.resume.resume_run",
                        lambda run_id, answer=None: called.update(run_id=run_id, answer=answer))
    monkeypatch.setattr(app, "_run_and_report", lambda *a, **k: called.update(tui=True))
    app._offer_inline_resume(_cost_question(), run_id="t-stub", run_dir=tmp_path,
                             runs_root=tmp_path, interactive=True)
    assert called == {"run_id": "t-stub", "answer": "accept"}  # plain fallback, no TUI


def test_prepare_pause_resume_applies_and_clears(monkeypatch, tmp_path):
    """prepare_pause_resume applies the answer, clears the pause, and returns
    (orch, resume kwargs) — reusing the same contract as `capo resume`."""
    import capo.persistence.resume as resume
    from capo.persistence.session_store import SessionState, SessionStore

    run_dir = tmp_path / "t-stub"
    (run_dir / "reports").mkdir(parents=True)
    (run_dir / "reports" / "pending_question.json").write_text(json.dumps(_cost_question()))
    store = SessionStore(run_dir)
    store.save(SessionState(run_id="t-stub", local_run_dir=str(run_dir), paused=True,
                            pause_reason="cost_accept_overrun",
                            pending_question_path="reports/pending_question.json",
                            created_at="2026-01-01T00:00:00", updated_at="2026-01-01T00:00:00"))

    monkeypatch.setattr(resume, "_LOCAL_FINE_TUNING_ROOT", tmp_path)
    monkeypatch.setattr(resume, "_build_resume_orchestrator", lambda state: "ORCH")

    out = resume.prepare_pause_resume("t-stub", "accept")
    assert out is not None
    orch, kwargs = out
    assert orch == "ORCH"
    assert kwargs["resume_from_pause"] is True and kwargs["run_id"] == "t-stub"
    assert kwargs["answer_artifact"] == "reports/cost_overrun_decision.json"
    # the decision artifact was written and the pause was cleared on disk
    assert (run_dir / "reports" / "cost_overrun_decision.json").exists()
    assert store.load().paused is False

    # not paused / no state → None (caller falls back to resume_run)
    store.update(paused=False)
    assert resume.prepare_pause_resume("t-stub", "accept") is None
    assert resume.prepare_pause_resume("nope", "accept") is None


def test_offer_inline_resume_defer_points_at_resume(monkeypatch, capsys, tmp_path):
    import capo.cli.app as app

    monkeypatch.setattr(app, "prompt_pending_answer", lambda q: None)  # user defers
    called = {"prepared": False}
    monkeypatch.setattr("capo.persistence.resume.prepare_pause_resume",
                        lambda *a, **k: called.update(prepared=True))
    app._offer_inline_resume(_cost_question(), run_id="t-stub", run_dir=tmp_path,
                             runs_root=tmp_path, interactive=True)
    assert called["prepared"] is False
    assert "capo resume t-stub" in capsys.readouterr().out


def test_run_console_help_lists_run_commands(capsys):
    from capo.cli.run_console import _print_help

    _print_help()
    out = capsys.readouterr().out
    for cmd in ("/health", "/status", "/abort", "/tune", "/config", "/history", "/quit"):
        assert cmd in out
    assert "/retune" not in out  # chat-only command is not offered mid-run


# === shared interaction router (one dispatcher for plain + full-screen views) ===


def _run_ctx(tmp_path):
    import threading

    from capo.cli.run_console import RunContext

    (tmp_path / "state.json").write_text(json.dumps({"current_phase": "training"}))
    return RunContext(run_id="r1", run_dir=tmp_path, runs_root=tmp_path,
                      stop_event=threading.Event(), start_time=0.0)


def test_dispatch_run_command_actions(tmp_path, capsys):
    from capo.cli.run_console import dispatch_run_command

    ctx = _run_ctx(tmp_path)
    # empty line is a no-op; a plain message is saved as a note and keeps going
    assert dispatch_run_command("", ctx) == "continue"
    assert dispatch_run_command("bump the eval frequency", ctx) == "continue"
    assert "bump the eval frequency" in (tmp_path / "user_notes.txt").read_text()
    # /status keeps the console open and reports the live phase
    assert dispatch_run_command("/status", ctx) == "continue"
    assert "training" in capsys.readouterr().out
    # /quit and bare quit/exit both detach (run keeps going)
    assert dispatch_run_command("/quit", ctx) == "detach"
    assert dispatch_run_command("quit", ctx) == "detach"


def test_dispatch_run_command_prefix_and_ambiguous(tmp_path, capsys):
    from capo.cli.run_console import dispatch_run_command

    ctx = _run_ctx(tmp_path)
    # a unique prefix runs the full command: /stat → /status (reports the phase)
    assert dispatch_run_command("/stat", ctx) == "continue"
    assert "training" in capsys.readouterr().out
    # a unique prefix that detaches still works: /qui → /quit
    assert dispatch_run_command("/qui", ctx) == "detach"
    # an ambiguous prefix runs nothing; it lists candidates and signals 'narrow'
    assert dispatch_run_command("/he", ctx) == "narrow"
    out = capsys.readouterr().out
    assert "/help" in out and "/health" in out and "type more" in out
    # a genuinely unknown slash line is still saved as a note (unchanged)
    assert dispatch_run_command("/zzz nope", ctx) == "continue"
    assert "/zzz nope" in (tmp_path / "user_notes.txt").read_text()


def test_dispatch_run_command_blocks_nested_ui(tmp_path, capsys):
    from capo.cli.run_console import dispatch_run_command

    ctx = _run_ctx(tmp_path)
    ctx.allow_nested_ui = False  # the full-screen owner forbids a nested /config editor
    assert dispatch_run_command("/config", ctx) == "continue"
    assert "capo config" in capsys.readouterr().out  # points at the post-run command instead


def test_dispatch_abort_arms_then_confirms(tmp_path, monkeypatch, capsys):
    import signal

    import capo.cli.run_console as rc
    from capo.cli.run_console import dispatch_run_command

    ctx = _run_ctx(tmp_path)
    kills: list = []
    monkeypatch.setattr(rc.os, "kill", lambda pid, sig: kills.append(sig))  # don't really SIGINT

    # first /abort only ARMS the confirmation — the run is untouched.
    assert dispatch_run_command("/abort", ctx) == "continue"
    assert ctx.abort_pending is True
    assert not ctx.abort_event.is_set() and not ctx.stop_event.is_set()
    assert kills == []
    assert "Stop this run" in capsys.readouterr().out  # the gentle warning was shown

    # typing y COMMITS: signal the orchestrator + report 'abort' to the caller.
    assert dispatch_run_command("y", ctx) == "abort"
    assert ctx.abort_pending is False
    assert ctx.abort_event.is_set() and ctx.stop_event.is_set()
    assert kills == [signal.SIGINT]


def test_dispatch_abort_arm_then_cancel(tmp_path, monkeypatch, capsys):
    import capo.cli.run_console as rc
    from capo.cli.run_console import dispatch_run_command

    ctx = _run_ctx(tmp_path)
    kills: list = []
    monkeypatch.setattr(rc.os, "kill", lambda pid, sig: kills.append(sig))

    assert dispatch_run_command("/abort", ctx) == "continue"          # arm
    # anything other than y cancels — the run keeps going, nothing signalled.
    assert dispatch_run_command("actually, no", ctx) == "continue"
    assert ctx.abort_pending is False
    assert not ctx.abort_event.is_set() and kills == []
    assert "keeping the run going" in capsys.readouterr().out.lower()


# === full-screen run view (run_view.RunConsole) ===


def test_run_view_feeds_and_renders(tmp_path):
    import threading

    from prompt_toolkit.formatted_text import to_formatted_text
    from rich.text import Text

    from capo.cli.run_view import RunConsole

    rc = RunConsole("run-abc", tmp_path, tmp_path, threading.Event(), 0.0)
    rc.feed(Text("12:00:00 [setup] writing train.py", style="green"))  # styled like real log lines
    assert len(rc._lines) == 1
    assert "\x1b[" in rc._lines[0]  # one log line stored as pre-rendered ANSI
    # the tail render is an ANSI formatted-text object carrying the line text
    logs = "".join(t for _s, t in to_formatted_text(rc._render_logs()))
    assert "writing train.py" in logs
    # command output captured into the pane lands as more lines
    with rc._capture_to_pane():
        from capo.cli.colors import console as shared

        shared.print("hello from a command")
    assert any("hello from a command" in ln for ln in rc._lines)


def test_run_view_scrollback(tmp_path):
    import threading

    from prompt_toolkit.formatted_text import to_formatted_text
    from rich.text import Text

    from capo.cli.run_view import RunConsole

    rc = RunConsole("run-scroll", tmp_path, tmp_path, threading.Event(), 0.0)
    for i in range(60):  # more than fits → there is history to scroll back to
        rc.feed(Text(f"line-{i:03d}", style="green"))

    def shown():
        return "".join(t for _s, t in to_formatted_text(rc._render_logs()))

    assert rc._scroll_top is None and "line-059" in shown()  # follows the live tail
    rc._scroll(-1)                                            # page up → off the tail
    assert rc._scroll_top is not None and "line-059" not in shown()
    rc._scroll(-1)                                            # keep paging up → the start
    assert "line-000" in shown()
    rc._scroll(+1)
    rc._scroll(+1)                                            # page down past bottom → live
    assert rc._scroll_top is None and "line-059" in shown()


def test_run_view_wheel_and_line_scroll(tmp_path):
    import threading

    from prompt_toolkit.formatted_text import to_formatted_text
    from rich.text import Text

    from capo.cli.run_view import RunConsole

    rc = RunConsole("run-wheel", tmp_path, tmp_path, threading.Event(), 0.0)
    for i in range(60):
        rc.feed(Text(f"line-{i:03d}", style="green"))

    def shown():
        return "".join(t for _s, t in to_formatted_text(rc._render_logs()))

    assert rc._scroll_top is None                  # following the live tail
    rc._wheel(-1)                                  # one wheel notch up → a few lines back
    assert rc._scroll_top is not None
    after_wheel = rc._scroll_top
    rc._scroll(-1, lines=1)                        # Shift+Up → exactly one line further up
    assert rc._scroll_top == after_wheel - 1
    for _ in range(40):                            # wheel back down past the bottom → live
        rc._wheel(+1)
    assert rc._scroll_top is None and "line-059" in shown()


def test_run_view_log_control_routes_wheel(tmp_path):
    from types import SimpleNamespace

    from prompt_toolkit.mouse_events import MouseEventType

    from capo.cli.run_view import _LogControl

    seen: list[int] = []
    ctrl = _LogControl(lambda: "", lambda d: seen.append(d))
    # wheel up / down are handled (return None) and step our scroller -1 / +1
    assert ctrl.mouse_handler(SimpleNamespace(event_type=MouseEventType.SCROLL_UP)) is None
    assert ctrl.mouse_handler(SimpleNamespace(event_type=MouseEventType.SCROLL_DOWN)) is None
    assert seen == [-1, 1]
    # any non-scroll event falls through (NotImplemented) so clicks/focus are untouched
    assert ctrl.mouse_handler(SimpleNamespace(event_type=MouseEventType.MOUSE_UP)) is NotImplemented


def test_run_view_scroll_keybindings(tmp_path):
    import threading

    from capo.cli.run_view import RunConsole

    rc = RunConsole("run-keys", tmp_path, tmp_path, threading.Event(), 0.0)
    bound = {getattr(k, "value", k) for b in rc._key_bindings().bindings for k in b.keys}
    # page (PgUp/PgDn) and line (Shift+↑/↓) scroll are both wired — the line keys
    # matter on laptops that have no dedicated PgUp/PgDn.
    assert {"pageup", "pagedown", "s-up", "s-down"} <= bound


def test_run_view_on_accept_narrow_keeps_buffer(tmp_path, monkeypatch):
    import threading

    from prompt_toolkit.buffer import Buffer

    import capo.cli.run_view as rv
    from capo.cli.run_view import RunConsole

    rc = RunConsole("run-narrow", tmp_path, tmp_path, threading.Event(), 0.0)
    # ambiguous prefix → the router returns 'narrow' → keep the line for more typing
    monkeypatch.setattr(rv, "dispatch_run_command", lambda text, ctx: "narrow")
    assert rc._on_accept(Buffer(multiline=False)) is True
    # a normal command clears the line (False) and snaps back to the live tail
    rc._scroll_top = 5
    monkeypatch.setattr(rv, "dispatch_run_command", lambda text, ctx: "continue")
    assert rc._on_accept(Buffer(multiline=False)) is False
    assert rc._scroll_top is None


def test_run_view_detach_streams_plaintext(tmp_path):
    import io
    import threading

    from rich.console import Console
    from rich.text import Text

    from capo.cli.colors import THEME
    from capo.cli.run_view import RunConsole

    rc = RunConsole("run-detach", tmp_path, tmp_path, threading.Event(), 0.0)
    buf = io.StringIO()  # stand in for the real terminal that _enter_plaintext_mode uses
    rc._plain_console = Console(theme=THEME, file=buf, force_terminal=True, color_system="truecolor")
    rc._plaintext = True
    rc.feed(Text("after detach", style="green"))
    assert "after detach" in buf.getvalue()  # streamed straight out, no ANSI leak
    assert len(rc._lines) == 0               # not buffered into the (closed) pane


def test_thinking_label_switches_after_15s():
    from capo.cli.chat import _thinking_label

    assert _thinking_label(0.0) == "thinking"
    assert _thinking_label(14.9) == "thinking"
    assert _thinking_label(15.0) == "still tuning"
    assert _thinking_label(120.0) == "still tuning"

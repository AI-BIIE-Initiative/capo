"""Integration tests for the orchestrator's cost-report wiring:
_build_cost_report assembles agent + infra lines, and _write_cost_report
persists run_cost.json + appends the Cost Report section to RUN_REPORT.md.
"""

from __future__ import annotations

import json

import pytest

from capo.report.cost import make_agent_cost, make_agent_cost_from_total
from capo.orchestration.fine_tuning_orchestrator import FineTuningOrchestrator


def _orch() -> FineTuningOrchestrator:
    return FineTuningOrchestrator(
        key_path="/tmp/nonexistent_key",
        ssh_key_name="k",
        model_id="facebook/esm2_t6_8M_UR50D",
        fine_tune_strategy="linear-probe",
        dataset_ref="org/ds",
        tolerance_threshold=0.1,
        enable_hf_research=False,
        enable_memory=False,
    )


def test_build_cost_report_combines_agent_and_infra():
    orch = _orch()
    orch._agent_cost_entries = [
        make_agent_cost("infrastructure", "claude-sonnet-4-6", sdk_cost_usd=0.10),
        make_agent_cost_from_total("training-health-monitor", "claude-haiku-4-5", 0.02),
    ]
    infra_data = {
        "instance_id": "i-1",
        "instance_type": "gpu_1x_a100",
        "resolved_gpu": "1x A100",
        "hourly_rate_usd": 1.49,
    }
    report = orch._build_cost_report(
        local_run_dir=None,  # _launched_at_iso tolerates a missing dir via exists()
        infra_data=infra_data,
        actual_cost_usd=0.75,
    )
    assert report.total_agent_cost_usd == pytest.approx(0.12)
    assert report.total_infra_cost_usd == pytest.approx(0.75)  # finalizer value wins
    assert report.total_cost_usd == pytest.approx(0.87)
    assert report.infra_costs[0].instance_id == "i-1"


def test_write_cost_report_appends_section(tmp_path):
    orch = _orch()
    orch._agent_cost_entries = [
        make_agent_cost("infrastructure", "claude-sonnet-4-6", sdk_cost_usd=0.10),
    ]
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    # Pretend the finalizer already wrote RUN_REPORT.md.
    report_md = tmp_path / "RUN_REPORT.md"
    report_md.write_text("# Run report\n\n## Results\n- ok\n", encoding="utf-8")

    report = orch._build_cost_report(tmp_path, {"hourly_rate_usd": 2.0, "instance_type": "x"}, 1.0)
    orch._write_cost_report(tmp_path, reports_dir, report)

    # run_cost.json persisted and parseable
    saved = json.loads((reports_dir / "run_cost.json").read_text(encoding="utf-8"))
    assert "total_cost_usd" in saved

    # Cost Report section appended exactly once
    body = report_md.read_text(encoding="utf-8")
    assert body.count("## Cost Report") == 1
    assert "### Agent Costs" in body

    # Idempotent: a second write does not duplicate the section
    orch._write_cost_report(tmp_path, reports_dir, report)
    assert report_md.read_text(encoding="utf-8").count("## Cost Report") == 1


def test_write_cost_report_no_run_report_still_writes_json(tmp_path):
    orch = _orch()
    orch._agent_cost_entries = []
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    report = orch._build_cost_report(tmp_path, None, None)
    orch._write_cost_report(tmp_path, reports_dir, report)
    assert (reports_dir / "run_cost.json").exists()
    assert not (tmp_path / "RUN_REPORT.md").exists()

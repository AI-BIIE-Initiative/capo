"""
Unit + integration tests for `capo.memory.run_report`.

Covers:
  - parse_frontmatter on a well-formed RUN_REPORT.md
  - validate_frontmatter (missing-required + unknown-key paths)
  - append_index_block: fresh write, dedup-by-run_id rewrite, concurrent locking
  - CLI: `python -m capo.memory.run_report append-from-report`
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import yaml

from capo.memory import run_report as rr


VALID_FRONTMATTER: dict = {
    "run_id": "ace2-esm2-20260611-1035-e4df",
    "task_summary": "Fine-tune ESM2-8M on ACE2 binding (linear-probe).",
    "modality": "protein",
    "target": "ACE2",
    "organism": "multi",
    "assay": "binding",
    "best_metric_name": "val_mcc",
    "best_metric_value": 0.78,
    "final_val_loss": 0.32,
    "key_decisions": ["Linear-probe sufficient — model_selection.json."],
    "key_findings": ["Cluster-aware split improves MCC by 0.12."],
    "key_pitfalls": ["OOM at batch 64; dropped to 32 + grad_accum=2."],
    "report_path": "capo/ace2-esm2-20260611-1035-e4df/RUN_REPORT.md",
}


def _write_report(run_dir: Path, frontmatter: dict, body: str = "# Run report\n\nBody.\n") -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    report = run_dir / "RUN_REPORT.md"
    yaml_block = yaml.safe_dump(frontmatter, sort_keys=False)
    report.write_text(f"---\n{yaml_block}---\n\n{body}", encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# parse_frontmatter
# ---------------------------------------------------------------------------


def test_parse_frontmatter_roundtrip(tmp_path: Path):
    report = _write_report(tmp_path / "run", VALID_FRONTMATTER)
    parsed = rr.parse_frontmatter(report)
    assert parsed is not None
    assert parsed["run_id"] == VALID_FRONTMATTER["run_id"]
    assert parsed["best_metric_value"] == pytest.approx(0.78)
    assert parsed["key_decisions"] == VALID_FRONTMATTER["key_decisions"]


def test_parse_frontmatter_missing_file(tmp_path: Path):
    assert rr.parse_frontmatter(tmp_path / "nope.md") is None


def test_parse_frontmatter_no_frontmatter(tmp_path: Path):
    p = tmp_path / "RUN_REPORT.md"
    p.write_text("# Just a markdown file with no frontmatter.\n", encoding="utf-8")
    assert rr.parse_frontmatter(p) is None


def test_parse_frontmatter_unclosed_block(tmp_path: Path):
    p = tmp_path / "RUN_REPORT.md"
    p.write_text("---\nrun_id: x\n# never closes\n", encoding="utf-8")
    assert rr.parse_frontmatter(p) is None


# ---------------------------------------------------------------------------
# validate_frontmatter
# ---------------------------------------------------------------------------


def test_validate_accepts_required_subset():
    rr.validate_frontmatter(VALID_FRONTMATTER)


def test_validate_rejects_missing_required():
    bad = dict(VALID_FRONTMATTER)
    del bad["run_id"]
    with pytest.raises(ValueError, match="missing required fields"):
        rr.validate_frontmatter(bad)


def test_validate_rejects_unknown_keys():
    bad = dict(VALID_FRONTMATTER)
    bad["extra_field"] = "drift"
    with pytest.raises(ValueError, match="unsupported fields"):
        rr.validate_frontmatter(bad)


# ---------------------------------------------------------------------------
# append_index_block
# ---------------------------------------------------------------------------


def test_append_index_block_fresh(tmp_path: Path):
    index = tmp_path / "runs_index.md"
    lock = tmp_path / ".runs_index.lock"
    rr.append_index_block(VALID_FRONTMATTER, index_path=index, lock_path=lock)
    assert index.exists()
    blocks = rr.read_index_blocks(index)
    assert len(blocks) == 1
    assert blocks[0]["run_id"] == VALID_FRONTMATTER["run_id"]


def test_append_index_block_dedup_by_run_id(tmp_path: Path):
    index = tmp_path / "runs_index.md"
    lock = tmp_path / ".runs_index.lock"
    rr.append_index_block(VALID_FRONTMATTER, index_path=index, lock_path=lock)
    # Re-append with the same run_id but updated metric — should replace, not duplicate.
    updated = dict(VALID_FRONTMATTER)
    updated["best_metric_value"] = 0.81
    rr.append_index_block(updated, index_path=index, lock_path=lock)
    blocks = rr.read_index_blocks(index)
    assert len(blocks) == 1
    assert blocks[0]["best_metric_value"] == pytest.approx(0.81)


def test_append_index_block_appends_distinct_runs(tmp_path: Path):
    index = tmp_path / "runs_index.md"
    lock = tmp_path / ".runs_index.lock"
    rr.append_index_block(VALID_FRONTMATTER, index_path=index, lock_path=lock)
    other = dict(VALID_FRONTMATTER, run_id="other-run-001", report_path="capo/other-run-001/RUN_REPORT.md")
    rr.append_index_block(other, index_path=index, lock_path=lock)
    blocks = rr.read_index_blocks(index)
    assert {b["run_id"] for b in blocks} == {VALID_FRONTMATTER["run_id"], "other-run-001"}


def test_append_index_block_concurrent_writers(tmp_path: Path):
    """Two threads write distinct run_ids concurrently; both should land cleanly."""
    index = tmp_path / "runs_index.md"
    lock = tmp_path / ".runs_index.lock"

    def write(rid: str):
        fm = dict(VALID_FRONTMATTER, run_id=rid, report_path=f"capo/{rid}/RUN_REPORT.md")
        rr.append_index_block(fm, index_path=index, lock_path=lock)

    threads = [threading.Thread(target=write, args=(f"run-{i:03d}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    blocks = rr.read_index_blocks(index)
    rids = {b["run_id"] for b in blocks}
    assert rids == {f"run-{i:03d}" for i in range(8)}, rids


def test_append_index_block_rejects_invalid_frontmatter(tmp_path: Path):
    index = tmp_path / "runs_index.md"
    lock = tmp_path / ".runs_index.lock"
    bad = dict(VALID_FRONTMATTER)
    del bad["task_summary"]
    with pytest.raises(ValueError):
        rr.append_index_block(bad, index_path=index, lock_path=lock)
    assert not index.exists()


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_cli_append_from_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run_dir = tmp_path / "my-run"
    _write_report(run_dir, VALID_FRONTMATTER)

    index = tmp_path / "runs_index.md"
    lock = tmp_path / ".runs_index.lock"
    monkeypatch.setattr(rr, "INDEX_PATH", index)
    monkeypatch.setattr(rr, "INDEX_LOCK_PATH", lock)

    rc = rr.main(["append-from-report", "--run-dir", str(run_dir)])
    assert rc == 0
    assert index.exists()
    blocks = rr.read_index_blocks(index)
    assert len(blocks) == 1
    assert blocks[0]["run_id"] == VALID_FRONTMATTER["run_id"]


def test_cli_append_from_report_missing_file(tmp_path: Path):
    rc = rr.main(["append-from-report", "--run-dir", str(tmp_path / "nope")])
    assert rc == 2


def test_cli_append_from_report_no_frontmatter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run_dir = tmp_path / "run-without-frontmatter"
    run_dir.mkdir(parents=True)
    (run_dir / "RUN_REPORT.md").write_text("# no frontmatter here\n", encoding="utf-8")
    monkeypatch.setattr(rr, "INDEX_PATH", tmp_path / "runs_index.md")
    monkeypatch.setattr(rr, "INDEX_LOCK_PATH", tmp_path / ".runs_index.lock")
    rc = rr.main(["append-from-report", "--run-dir", str(run_dir)])
    assert rc == 3


def test_cli_append_from_report_invalid_frontmatter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run_dir = tmp_path / "bad-run"
    bad = dict(VALID_FRONTMATTER)
    del bad["report_path"]
    _write_report(run_dir, bad)
    monkeypatch.setattr(rr, "INDEX_PATH", tmp_path / "runs_index.md")
    monkeypatch.setattr(rr, "INDEX_LOCK_PATH", tmp_path / ".runs_index.lock")
    rc = rr.main(["append-from-report", "--run-dir", str(run_dir)])
    assert rc == 4

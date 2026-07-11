"""Tests for single-instance / reuse plumbing:
- SessionState.active_instance_id + reuse_existing_instance round-trip
- _resolve_reuse_existing precedence (non-interactive reuse default)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from capo.persistence.session_store import SessionStore, new_session

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_run_ft():
    spec = importlib.util.spec_from_file_location(
        "run_fine_tuning_under_test", _REPO_ROOT / "scripts" / "run_fine_tuning.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# SessionState new fields
# ---------------------------------------------------------------------------

def test_new_session_defaults(tmp_path):
    state = new_session(
        run_id="r1",
        local_run_dir=tmp_path,
        remote_run_dir="~/capo_runs/r1",
        tolerance_threshold=0.1,
    )
    assert state.active_instance_id is None
    assert state.reuse_existing_instance is True


def test_active_instance_id_round_trip(tmp_path):
    store = SessionStore(tmp_path)
    store.save(
        new_session(
            run_id="r1",
            local_run_dir=tmp_path,
            remote_run_dir="~/capo_runs/r1",
            tolerance_threshold=0.0,
            reuse_existing_instance=False,
        )
    )
    store.update(active_instance_id="i-xyz")
    loaded = store.load()
    assert loaded is not None
    assert loaded.active_instance_id == "i-xyz"
    assert loaded.reuse_existing_instance is False


# ---------------------------------------------------------------------------
# _resolve_reuse_existing precedence
# ---------------------------------------------------------------------------

def test_reuse_default_true():
    mod = _load_run_ft()
    assert mod._resolve_reuse_existing({}) is True


def test_reuse_allow_reuse_existing_wins():
    mod = _load_run_ft()
    assert mod._resolve_reuse_existing({"allow_reuse_existing": False}) is False


def test_reuse_alias_reuse_existing_instance():
    mod = _load_run_ft()
    assert mod._resolve_reuse_existing({"reuse_existing_instance": False}) is False


def test_reuse_infra_block_alias():
    mod = _load_run_ft()
    assert mod._resolve_reuse_existing({"infra": {"reuse_existing_instance": False}}) is False


def test_reuse_primary_key_precedence_over_alias():
    mod = _load_run_ft()
    cfg = {"allow_reuse_existing": True, "reuse_existing_instance": False}
    assert mod._resolve_reuse_existing(cfg) is True

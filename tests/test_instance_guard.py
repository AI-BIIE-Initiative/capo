"""Tests for the single-Lambda-instance-per-run guard."""

from __future__ import annotations

import pytest

from capo.remote import instance_guard
from capo.remote.instance_guard import SingleInstanceViolation


@pytest.fixture(autouse=True)
def _clean_guard(monkeypatch):
    # Isolate each test: clear CAPO_RUN_ID and reset the module state.
    monkeypatch.delenv("CAPO_RUN_ID", raising=False)
    instance_guard.reset()
    yield
    instance_guard.reset()


def test_first_claim_succeeds():
    instance_guard.assert_can_provision()  # nothing claimed yet
    instance_guard.claim("i-aaa")
    assert instance_guard.active_instance_id() == "i-aaa"


def test_second_distinct_provision_blocked():
    instance_guard.claim("i-aaa")
    with pytest.raises(SingleInstanceViolation):
        instance_guard.assert_can_provision()


def test_claiming_same_instance_is_idempotent():
    instance_guard.claim("i-aaa")
    instance_guard.claim("i-aaa")  # no raise — same instance re-attached
    assert instance_guard.active_instance_id() == "i-aaa"


def test_claiming_different_instance_raises():
    instance_guard.claim("i-aaa")
    with pytest.raises(SingleInstanceViolation):
        instance_guard.claim("i-bbb")


def test_multi_gpu_single_instance_is_one_claim():
    # A multi-GPU instance is still a single provision/claim.
    instance_guard.assert_can_provision()
    instance_guard.claim("i-8xh100")
    assert instance_guard.active_instance_id() == "i-8xh100"
    # No second provision is permitted.
    with pytest.raises(SingleInstanceViolation):
        instance_guard.assert_can_provision()


def test_reset_clears_state():
    instance_guard.claim("i-aaa")
    instance_guard.reset()
    assert instance_guard.active_instance_id() is None
    instance_guard.assert_can_provision()  # allowed again


def test_run_id_change_auto_resets(monkeypatch):
    monkeypatch.setenv("CAPO_RUN_ID", "run-1")
    instance_guard.claim("i-aaa")
    assert instance_guard.active_instance_id() == "i-aaa"
    # New run id → state forgotten, provisioning allowed again.
    monkeypatch.setenv("CAPO_RUN_ID", "run-2")
    instance_guard.assert_can_provision()
    instance_guard.claim("i-bbb")
    assert instance_guard.active_instance_id() == "i-bbb"


def test_claim_none_is_noop():
    instance_guard.claim(None)
    assert instance_guard.active_instance_id() is None

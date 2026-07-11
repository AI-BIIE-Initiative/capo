"""Tests for the pure tool functions in capo.mcp.tools.lambda_tools.

These tests do not import FastMCP. They exercise the dict-returning tool
functions directly, with the underlying capo.remote helpers patched out.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from capo.mcp.tools import lambda_tools as t
from capo.remote.lambda_session import LambdaInstance


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _make_instance(**overrides) -> LambdaInstance:
    base = dict(
        instance_id="i-abc",
        ip="1.2.3.4",
        region="us-west-1",
        instance_type="gpu_1x_a100",
        status="active",
        ssh_key_names=["lambda_main"],
        name="ace2-finetune",
        price_cents_per_hour=149,
        price_dollars_per_hour=1.49,
        launched_at=_iso(datetime.now(timezone.utc) - timedelta(hours=1)),
    )
    base.update(overrides)
    return LambdaInstance(**base)


# ---------------------------------------------------------------------------
# lambda_provision_instance
# ---------------------------------------------------------------------------

def test_lambda_provision_instance_serialization():
    inst = _make_instance(status="booting", ip=None)
    with patch("capo.mcp.tools.lambda_tools.provision_instance", return_value=inst):
        result = t.lambda_provision_instance(
            instance_type="gpu_1x_a100",
            ssh_key_name="lambda_main",
            region="us-west-1",
        )
    assert result["ok"] is True
    assert result["instance_id"] == "i-abc"
    assert result["status"] == "booting"
    assert result["price_dollars_per_hour"] == pytest.approx(1.49)
    # Whole result is JSON-serialisable
    json.dumps(result)


def test_lambda_provision_instance_error():
    with patch(
        "capo.mcp.tools.lambda_tools.provision_instance",
        side_effect=RuntimeError("boom"),
    ):
        result = t.lambda_provision_instance(
            instance_type="gpu_1x_a100",
            ssh_key_name="lambda_main",
        )
    assert result["ok"] is False
    assert "boom" in result["error"]


# ---------------------------------------------------------------------------
# Cost estimate tools
# ---------------------------------------------------------------------------

def test_lambda_get_cost_estimate_serialization():
    inst = _make_instance(
        launched_at=_iso(datetime.now(timezone.utc) - timedelta(minutes=30))
    )
    with patch("capo.mcp.tools.lambda_tools.get_instance", return_value=inst):
        result = t.lambda_get_cost_estimate(
            instance_id="i-abc",
            budget_limit_dollars=10.0,
            budget_warning_threshold_dollars=5.0,
        )
    assert result["ok"] is True
    assert result["instance_id"] == "i-abc"
    estimate = result["estimate"]
    # Every LambdaCostEstimate field is present and round-trips through JSON
    serialised = json.dumps(estimate)
    assert "estimated_cost_dollars" in serialised
    assert "elapsed_hours" in serialised
    assert estimate["budget_limit_dollars"] == 10.0
    assert estimate["budget_warning_threshold_dollars"] == 5.0
    assert estimate["over_budget"] is False
    assert estimate["estimated_cost_dollars"] == pytest.approx(0.5 * 1.49, abs=0.1)


def test_lambda_get_first_cost_estimate_t0():
    inst = _make_instance(
        status="booting",
        launched_at=_iso(datetime.now(timezone.utc)),
    )
    with patch("capo.mcp.tools.lambda_tools.get_instance", return_value=inst):
        result = t.lambda_get_first_cost_estimate(instance_id="i-abc")
    assert result["ok"] is True
    assert result["estimate"]["estimated_cost_dollars"] == pytest.approx(0.0, abs=0.05)


def test_lambda_get_cost_estimate_handles_error():
    with patch(
        "capo.mcp.tools.lambda_tools.get_instance",
        side_effect=RuntimeError("api down"),
    ):
        result = t.lambda_get_cost_estimate(instance_id="i-abc")
    assert result["ok"] is False
    assert "api down" in result["error"]


# ---------------------------------------------------------------------------
# lambda_terminate_safe
# ---------------------------------------------------------------------------

def test_lambda_terminate_safe_rejects_mismatch():
    inst = _make_instance(ssh_key_names=["other_user_key"])
    with patch("capo.mcp.tools.lambda_tools.safe_terminate_instance") as fake_safe:
        fake_safe.side_effect = PermissionError(
            "Refusing to terminate i-abc: instance ssh_key_names=['other_user_key']"
        )
        result = t.lambda_terminate_safe(
            instance_id="i-abc",
            expected_ssh_key_names=["mine"],
        )
    assert result["ok"] is False
    assert result.get("reason") == "ownership_mismatch"
    assert "Refusing to terminate" in result["error"]


def test_lambda_terminate_safe_proceeds_on_match():
    inst = _make_instance(ssh_key_names=["mine"])
    with patch(
        "capo.mcp.tools.lambda_tools.safe_terminate_instance",
        return_value=inst,
    ) as fake_safe:
        result = t.lambda_terminate_safe(
            instance_id="i-abc",
            expected_ssh_key_names=["mine"],
        )
    assert result["ok"] is True
    assert result["instance_id"] == "i-abc"
    assert result["verified_ssh_key_names"] == ["mine"]
    fake_safe.assert_called_once()


# ---------------------------------------------------------------------------
# lambda_preflight tool dispatch
# ---------------------------------------------------------------------------

def test_lambda_preflight_dispatches_to_runtime():
    fake_payload = {"ok": True, "checks": [{"name": "x", "passed": True, "detail": ""}]}
    with patch("capo.mcp.tools.lambda_tools.run_preflight", return_value=fake_payload):
        result = t.lambda_preflight()
    assert result == fake_payload
    json.dumps(result)


def test_lambda_preflight_runtime_failure_caught():
    with patch(
        "capo.mcp.tools.lambda_tools.run_preflight",
        side_effect=RuntimeError("network"),
    ):
        result = t.lambda_preflight()
    assert result["ok"] is False
    assert "network" in result["error"]
    assert result["checks"] == []


# ---------------------------------------------------------------------------
# Discovery tools — thin wrappers, just check shape
# ---------------------------------------------------------------------------

def test_lambda_find_local_ssh_keys_wrapper():
    fake_keys = [
        {"path": "/home/me/.ssh/k", "type": "ed25519", "has_pub": True, "permissions_ok": True}
    ]
    with patch(
        "capo.mcp.tools.lambda_tools.find_local_ssh_keys",
        return_value=fake_keys,
    ):
        result = t.lambda_find_local_ssh_keys(ssh_dir="/home/me/.ssh")
    assert result["ok"] is True
    assert result["count"] == 1
    assert result["keys"] == fake_keys


def test_lambda_list_instance_types_wrapper():
    fake_types = [
        {"name": "gpu_1x_a100", "price_dollars_per_hour": 1.49,
         "regions_with_capacity_available": ["us-west-1"], "specs": {}}
    ]
    with patch(
        "capo.mcp.tools.lambda_tools.list_instance_types",
        return_value=fake_types,
    ):
        result = t.lambda_list_instance_types()
    assert result["ok"] is True
    assert result["instance_types"] == fake_types


# ---------------------------------------------------------------------------
# CANONICAL_TOOLS registry — every entry maps to the function exported
# ---------------------------------------------------------------------------

def test_canonical_tools_unique_and_callable():
    names = [n for n, _ in t.CANONICAL_TOOLS]
    assert len(names) == len(set(names)), "duplicate tool names"
    for name, fn in t.CANONICAL_TOOLS:
        assert callable(fn), f"{name} is not callable"


def test_canonical_tools_includes_all_user_listed_canonicals():
    expected = {
        "lambda_find_local_ssh_keys",
        "lambda_list_ssh_keys",
        "lambda_list_instance_types",
        "lambda_preflight",
        "lambda_provision_instance",
        "lambda_get_first_cost_estimate",
        "lambda_get_cost_estimate",
        "lambda_start_session",
        "lambda_run_command",
        "lambda_push_files",
        "lambda_pull_files",
        "lambda_terminate_safe",
    }
    actual = {n for n, _ in t.CANONICAL_TOOLS}
    missing = expected - actual
    assert not missing, f"missing canonical tools: {missing}"

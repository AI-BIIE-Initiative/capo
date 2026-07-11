"""Tests for lambda_pricing — parse_instance pricing fields, get_instance_type_price,
estimate_cost (elapsed × rate, with budget thresholds)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from capo.remote.lambda_pricing import (
    LambdaCostEstimate,
    estimate_cost,
    get_instance_type_price,
    parse_datetime,
)
from capo.remote.lambda_session import parse_instance


# ---------------------------------------------------------------------------
# parse_instance pricing
# ---------------------------------------------------------------------------

def test_parse_instance_with_pricing():
    payload = {
        "id": "i-abc",
        "ip": "1.2.3.4",
        "name": "ace2-finetune",
        "status": "active",
        "region": {"name": "us-west-1"},
        "instance_type": {"name": "gpu_1x_a100", "price_cents_per_hour": 149},
        "ssh_key_names": [{"name": "lambda_main"}],
        "created_at": "2026-05-07T12:00:00Z",
    }
    inst = parse_instance(payload)
    assert inst.instance_id == "i-abc"
    assert inst.ip == "1.2.3.4"
    assert inst.region == "us-west-1"
    assert inst.instance_type == "gpu_1x_a100"
    assert inst.status == "active"
    assert inst.name == "ace2-finetune"
    assert inst.price_cents_per_hour == 149
    assert inst.price_dollars_per_hour == pytest.approx(1.49)
    assert inst.launched_at == "2026-05-07T12:00:00Z"
    assert inst.ssh_key_names == ["lambda_main"]


def test_parse_instance_missing_pricing():
    payload = {
        "id": "i-xyz",
        "ip": None,
        "status": "booting",
        "region": "us-east-1",
        "instance_type": "gpu_1x_a10",
        "ssh_key_names": ["lambda_main"],
    }
    inst = parse_instance(payload)
    assert inst.price_cents_per_hour is None
    assert inst.price_dollars_per_hour is None
    assert inst.launched_at is None
    assert inst.name is None
    assert inst.region == "us-east-1"
    assert inst.instance_type == "gpu_1x_a10"


def test_parse_instance_string_ssh_keys():
    payload = {
        "id": "i-1",
        "status": "active",
        "ssh_key_names": ["k1", "k2"],
    }
    inst = parse_instance(payload)
    assert inst.ssh_key_names == ["k1", "k2"]


def test_parse_instance_alternate_timestamp_field():
    payload = {"id": "i-1", "status": "active", "launched_at": "2026-01-01T00:00:00Z"}
    inst = parse_instance(payload)
    assert inst.launched_at == "2026-01-01T00:00:00Z"


# ---------------------------------------------------------------------------
# get_instance_type_price
# ---------------------------------------------------------------------------

def test_get_instance_type_price_hit():
    catalog = {
        "gpu_1x_a100": {"instance_type": {"price_cents_per_hour": 149}},
        "gpu_8x_h100": {"instance_type": {"price_cents_per_hour": 2592}},
    }
    assert get_instance_type_price(catalog, "gpu_1x_a100") == pytest.approx(1.49)
    assert get_instance_type_price(catalog, "gpu_8x_h100") == pytest.approx(25.92)


def test_get_instance_type_price_miss():
    catalog = {"gpu_1x_a100": {"instance_type": {"price_cents_per_hour": 149}}}
    assert get_instance_type_price(catalog, "nonexistent") is None


def test_get_instance_type_price_missing_price_field():
    catalog = {"gpu_1x_a100": {"instance_type": {}}}
    assert get_instance_type_price(catalog, "gpu_1x_a100") is None


# ---------------------------------------------------------------------------
# parse_datetime
# ---------------------------------------------------------------------------

def test_parse_datetime_z_suffix():
    dt = parse_datetime("2026-05-07T12:00:00Z")
    assert dt is not None
    assert dt.tzinfo is not None
    assert dt.year == 2026 and dt.month == 5 and dt.day == 7


def test_parse_datetime_offset():
    dt = parse_datetime("2026-05-07T12:00:00+00:00")
    assert dt is not None and dt.tzinfo is not None


def test_parse_datetime_invalid():
    assert parse_datetime("not-a-date") is None


# ---------------------------------------------------------------------------
# estimate_cost
# ---------------------------------------------------------------------------

def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def test_estimate_cost_basic():
    started = datetime.now(timezone.utc) - timedelta(minutes=90)
    result = estimate_cost(
        instance_id="i-abc",
        price_dollars_per_hour=1.49,
        started_at=_iso(started),
    )
    assert isinstance(result, LambdaCostEstimate)
    assert result.elapsed_hours == pytest.approx(1.5, abs=0.05)
    assert result.estimated_cost_dollars == pytest.approx(1.5 * 1.49, abs=0.1)
    assert result.over_budget is False
    assert result.over_warning_threshold is False


def test_estimate_cost_t0():
    now = datetime.now(timezone.utc)
    result = estimate_cost(
        instance_id="i-abc",
        price_dollars_per_hour=1.49,
        started_at=_iso(now),
    )
    assert result.elapsed_hours == pytest.approx(0.0, abs=0.01)
    assert result.estimated_cost_dollars == pytest.approx(0.0, abs=0.02)


def test_estimate_cost_budget_thresholds_trip():
    started = datetime.now(timezone.utc) - timedelta(hours=2)
    result = estimate_cost(
        instance_id="i-abc",
        price_dollars_per_hour=1.49,
        started_at=_iso(started),
        budget_limit_dollars=2.0,
        budget_warning_threshold_dollars=1.0,
    )
    # 2h × 1.49 ≈ 2.98 → over both
    assert result.over_warning_threshold is True
    assert result.over_budget is True


def test_estimate_cost_budget_thresholds_clear():
    started = datetime.now(timezone.utc) - timedelta(minutes=10)
    result = estimate_cost(
        instance_id="i-abc",
        price_dollars_per_hour=1.49,
        started_at=_iso(started),
        budget_limit_dollars=10.0,
        budget_warning_threshold_dollars=5.0,
    )
    assert result.over_warning_threshold is False
    assert result.over_budget is False


def test_estimate_cost_missing_price():
    result = estimate_cost(
        instance_id="i-abc",
        price_dollars_per_hour=None,
        started_at="2026-05-07T12:00:00Z",
    )
    assert result.elapsed_hours is None
    assert result.estimated_cost_dollars is None


def test_estimate_cost_missing_started_at():
    result = estimate_cost(
        instance_id="i-abc",
        price_dollars_per_hour=1.49,
        started_at=None,
    )
    assert result.elapsed_hours is None
    assert result.estimated_cost_dollars is None


def test_estimate_cost_unparseable_started_at():
    result = estimate_cost(
        instance_id="i-abc",
        price_dollars_per_hour=1.49,
        started_at="garbage-timestamp",
    )
    assert result.elapsed_hours is None
    assert result.estimated_cost_dollars is None

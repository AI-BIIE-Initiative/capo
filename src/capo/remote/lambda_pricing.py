"""Pricing and cost-estimate primitives for Lambda Cloud GPU instances.

Pure data + arithmetic — no I/O. The caller is responsible for fetching
price_dollars_per_hour and started_at (typically from
capo.remote.get_instance which surfaces both fields via
capo.remote.parse_instance).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class LambdaCostEstimate:
    instance_id: str
    price_dollars_per_hour: float | None
    started_at: str | None
    observed_at: str
    elapsed_hours: float | None
    estimated_cost_dollars: float | None
    budget_limit_dollars: float | None = None
    budget_warning_threshold_dollars: float | None = None
    over_warning_threshold: bool = False
    over_budget: bool = False


def parse_datetime(value: str) -> datetime | None:
    """Parse an ISO-8601 timestamp into a tz-aware datetime, or None.

    Tolerates the Z suffix that some Lambda API payloads use in place of
    +00:00. Naive timestamps are assumed UTC.
    """
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def get_instance_type_price(
    instance_types: dict[str, Any],
    instance_type_name: str,
) -> float | None:
    """Return the per-hour price in dollars for instance_type_name, or None.

    instance_types is the catalog payload from
    GET /api/v1/instance-types keyed by instance-type name (Lambda's standard
    response shape).
    """
    item = instance_types.get(instance_type_name)
    if not item:
        return None

    type_info = item.get("instance_type", {})
    price_cents = type_info.get("price_cents_per_hour")

    if price_cents is None:
        return None

    return price_cents / 100


def estimate_cost(
    *,
    instance_id: str,
    price_dollars_per_hour: float | None,
    started_at: str | None,
    budget_limit_dollars: float | None = None,
    budget_warning_threshold_dollars: float | None = None,
) -> LambdaCostEstimate:
    """Compute elapsed-time × hourly-rate cost for one instance.

    Returns a :class:`LambdaCostEstimate` whose numeric fields are None when
    the inputs are insufficient (no price or no start time). Budget thresholds
    are passed through to the result and toggle the over_* flags when the
    estimate has crossed them.
    """
    observed_dt = datetime.now(timezone.utc)
    observed_at = observed_dt.isoformat()

    if price_dollars_per_hour is None or started_at is None:
        return LambdaCostEstimate(
            instance_id=instance_id,
            price_dollars_per_hour=price_dollars_per_hour,
            started_at=started_at,
            observed_at=observed_at,
            elapsed_hours=None,
            estimated_cost_dollars=None,
            budget_limit_dollars=budget_limit_dollars,
            budget_warning_threshold_dollars=budget_warning_threshold_dollars,
        )

    start_dt = parse_datetime(started_at)
    if start_dt is None:
        return LambdaCostEstimate(
            instance_id=instance_id,
            price_dollars_per_hour=price_dollars_per_hour,
            started_at=started_at,
            observed_at=observed_at,
            elapsed_hours=None,
            estimated_cost_dollars=None,
            budget_limit_dollars=budget_limit_dollars,
            budget_warning_threshold_dollars=budget_warning_threshold_dollars,
        )

    elapsed_hours = max(0.0, (observed_dt - start_dt).total_seconds() / 3600)
    estimated_cost = elapsed_hours * price_dollars_per_hour

    return LambdaCostEstimate(
        instance_id=instance_id,
        price_dollars_per_hour=price_dollars_per_hour,
        started_at=started_at,
        observed_at=observed_at,
        elapsed_hours=elapsed_hours,
        estimated_cost_dollars=estimated_cost,
        budget_limit_dollars=budget_limit_dollars,
        budget_warning_threshold_dollars=budget_warning_threshold_dollars,
        over_warning_threshold=(
            budget_warning_threshold_dollars is not None
            and estimated_cost >= budget_warning_threshold_dollars
        ),
        over_budget=(
            budget_limit_dollars is not None
            and estimated_cost >= budget_limit_dollars
        ),
    )

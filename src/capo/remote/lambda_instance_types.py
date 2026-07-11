"""Lambda Cloud instance-type catalog with live capacity and pricing."""
from __future__ import annotations

from typing import Any

import requests

from capo.remote.lambda_session import _lambda_api_key

LAMBDA_API_BASE = "https://cloud.lambdalabs.com/api/v1"


def list_instance_types(
    *,
    available_only: bool = True,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """GET /instance-types — return the catalog with capacity and pricing.

    Each item is normalized to::

        {
            "name": str,
            "price_cents_per_hour": int | None,
            "price_dollars_per_hour": float | None,
            "regions_with_capacity_available": list[str],
            "specs": dict,
        }

    When available_only is True (the default), entries with no live
    regional capacity are filtered out.
    """
    key = _lambda_api_key(api_key)
    resp = requests.get(
        f"{LAMBDA_API_BASE}/instance-types",
        auth=(key, ""),
        timeout=15,
    )
    resp.raise_for_status()

    payload = resp.json().get("data", {}) or {}
    out: list[dict[str, Any]] = []
    for type_name, item in payload.items():
        type_info = item.get("instance_type", {}) or {}
        price_cents = type_info.get("price_cents_per_hour")
        price_dollars = (
            price_cents / 100 if isinstance(price_cents, (int, float)) else None
        )

        regions_payload = item.get("regions_with_capacity_available", []) or []
        regions = [
            r.get("name", "") if isinstance(r, dict) else str(r)
            for r in regions_payload
        ]

        if available_only and not regions:
            continue

        out.append(
            {
                "name": type_info.get("name", type_name),
                "price_cents_per_hour": price_cents,
                "price_dollars_per_hour": price_dollars,
                "regions_with_capacity_available": regions,
                "specs": type_info.get("specs", {}) or {},
            }
        )

    return out

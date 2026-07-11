"""Pre-launch checks: env, CLI tools, key file permissions, API authentication."""
from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path
from typing import Any

import requests

from capo.remote.lambda_ssh_keys import _classify_header

LAMBDA_API_BASE = "https://cloud.lambdalabs.com/api/v1"


def _check(name: str, passed: bool, detail: str = "") -> dict[str, Any]:
    return {"name": name, "passed": passed, "detail": detail}


def _check_api_authenticates(api_key: str | None) -> dict[str, Any]:
    """Cheap GET /ssh-keys to confirm the key is valid."""
    if not api_key:
        return _check("api_authenticates", False, "no API key to test")
    try:
        resp = requests.get(
            f"{LAMBDA_API_BASE}/ssh-keys",
            auth=(api_key, ""),
            timeout=10,
        )
        if resp.status_code == 200:
            return _check("api_authenticates", True, "GET /ssh-keys → 200")
        return _check(
            "api_authenticates",
            False,
            f"GET /ssh-keys → {resp.status_code}",
        )
    except Exception as exc:
        return _check("api_authenticates", False, f"request failed: {exc}")


def run_preflight(
    key_path: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run preflight checks. Return ``{"ok": bool, "checks": [...]}``.

    The ``ok`` flag is the AND of every check's ``passed``.
    """
    resolved_api_key = api_key or os.environ.get("LAMBDA_API_KEY") or None

    checks: list[dict[str, Any]] = []

    checks.append(
        _check(
            "lambda_api_key_set",
            bool(resolved_api_key),
            "LAMBDA_API_KEY env var present" if resolved_api_key else "LAMBDA_API_KEY missing",
        )
    )

    checks.append(_check_api_authenticates(resolved_api_key))

    for tool in ("ssh", "rsync", "tmux"):
        path = shutil.which(tool)
        checks.append(
            _check(
                f"{tool}_available",
                path is not None,
                f"found at {path}" if path else f"{tool} not on PATH",
            )
        )

    if key_path:
        kp = Path(key_path).expanduser()
        if not kp.exists():
            checks.append(_check("ssh_key_exists", False, f"{kp} not found"))
        else:
            checks.append(_check("ssh_key_exists", True, str(kp)))
            mode = stat.S_IMODE(kp.stat().st_mode)
            permissions_ok = (mode & 0o077) == 0
            checks.append(
                _check(
                    "ssh_key_permissions",
                    permissions_ok,
                    f"mode {oct(mode)}{'' if permissions_ok else ' — should be 0600'}",
                )
            )
            header_type = _classify_header(kp)
            checks.append(
                _check(
                    "ssh_key_header",
                    header_type is not None,
                    f"detected: {header_type}" if header_type else "header not recognized",
                )
            )

    ok = all(c["passed"] for c in checks)
    return {"ok": ok, "checks": checks}

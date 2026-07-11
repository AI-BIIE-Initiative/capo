"""SSH-key discovery helpers for Lambda Cloud GPU connections.

``find_local_ssh_keys`` scans a local directory for likely private keys and
classifies them by file header — **never** reads or returns key material.

``list_remote_ssh_keys`` queries the Lambda Cloud REST API for keys registered
on the account.
"""
from __future__ import annotations

import stat
from pathlib import Path
from typing import Any

import requests

from capo.remote.lambda_session import _lambda_api_key

LAMBDA_API_BASE = "https://cloud.lambdalabs.com/api/v1"


_HEADER_TYPES: dict[str, str] = {
    "-----BEGIN OPENSSH PRIVATE KEY-----": "openssh",
    "-----BEGIN RSA PRIVATE KEY-----": "rsa",
    "-----BEGIN DSA PRIVATE KEY-----": "dsa",
    "-----BEGIN EC PRIVATE KEY-----": "ecdsa",
    "-----BEGIN ED25519 PRIVATE KEY-----": "ed25519",
}


def _classify_header(path: Path) -> str | None:
    """Return the key type derived from the first line of the file, or None.

    Reads only the first line — never parses key material.
    """
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline().strip()
    except OSError:
        return None
    return _HEADER_TYPES.get(first_line)


def _refine_openssh_type(path: Path) -> str:
    """For OpenSSH-format private keys, infer the algorithm from the filename.

    The OpenSSH header is the same for ed25519, rsa, ecdsa, etc. We do not
    parse the key body, so we fall back to filename-based hints.
    """
    name = path.name.lower()
    for token in ("ed25519", "ecdsa", "rsa", "dsa"):
        if token in name:
            return token
    return "openssh"


def find_local_ssh_keys(ssh_dir: Path | str | None = None) -> list[dict[str, Any]]:
    """Scan ``ssh_dir`` (default ``~/.ssh``) for likely private keys.

    Returns a list of ``{"path", "type", "has_pub", "permissions_ok"}``. Files
    whose first line does not match a known SSH private-key header are skipped.
    No key material is ever read or returned.
    """
    base = Path(ssh_dir).expanduser() if ssh_dir else Path.home() / ".ssh"
    if not base.is_dir():
        return []

    results: list[dict[str, Any]] = []
    for entry in sorted(base.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix == ".pub":
            continue
        if entry.name in {"known_hosts", "config", "authorized_keys"}:
            continue

        header_type = _classify_header(entry)
        if header_type is None:
            continue

        key_type = (
            _refine_openssh_type(entry) if header_type == "openssh" else header_type
        )

        try:
            mode = stat.S_IMODE(entry.stat().st_mode)
        except OSError:
            mode = 0o000

        permissions_ok = (mode & 0o077) == 0

        results.append(
            {
                "path": str(entry),
                "type": key_type,
                "has_pub": entry.with_suffix(entry.suffix + ".pub").exists()
                or Path(str(entry) + ".pub").exists(),
                "permissions_ok": permissions_ok,
            }
        )
    return results


def list_remote_ssh_keys(api_key: str | None = None) -> list[dict[str, Any]]:
    """GET /ssh-keys — return SSH keys registered on the Lambda account.

    Each result is ``{"name", "id", "public_key_fingerprint"}``. Private key
    material is never present in this response.
    """
    key = _lambda_api_key(api_key)
    resp = requests.get(
        f"{LAMBDA_API_BASE}/ssh-keys",
        auth=(key, ""),
        timeout=15,
    )
    resp.raise_for_status()

    data = resp.json().get("data", [])
    return [
        {
            "name": item.get("name", ""),
            "id": item.get("id", ""),
            "public_key_fingerprint": item.get("public_key_fingerprint", ""),
        }
        for item in data
    ]

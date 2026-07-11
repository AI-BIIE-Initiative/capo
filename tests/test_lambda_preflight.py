"""Tests for run_preflight: env, CLI tools, key file validity, API auth."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from capo.remote.lambda_preflight import run_preflight


def _check_named(checks, name):
    for c in checks:
        if c["name"] == name:
            return c
    raise AssertionError(f"check {name} not in {[c['name'] for c in checks]}")


def test_preflight_missing_api_key(monkeypatch):
    monkeypatch.delenv("LAMBDA_API_KEY", raising=False)
    result = run_preflight(api_key=None)
    assert result["ok"] is False
    api_check = _check_named(result["checks"], "lambda_api_key_set")
    assert api_check["passed"] is False


def test_preflight_full_pass(monkeypatch, tmp_path):
    monkeypatch.setenv("LAMBDA_API_KEY", "test-key")

    key = tmp_path / "lambda_main"
    key.write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nbody\n")
    os.chmod(key, 0o600)

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.raise_for_status.return_value = None

    with patch("capo.remote.lambda_preflight.shutil.which", return_value="/usr/bin/x"), \
         patch("capo.remote.lambda_preflight.requests") as fake_requests:
        fake_requests.get.return_value = fake_resp
        result = run_preflight(key_path=str(key))

    assert result["ok"] is True, result["checks"]
    assert _check_named(result["checks"], "lambda_api_key_set")["passed"] is True
    assert _check_named(result["checks"], "api_authenticates")["passed"] is True
    assert _check_named(result["checks"], "ssh_available")["passed"] is True
    assert _check_named(result["checks"], "ssh_key_exists")["passed"] is True
    assert _check_named(result["checks"], "ssh_key_permissions")["passed"] is True


def test_preflight_missing_rsync(monkeypatch):
    monkeypatch.setenv("LAMBDA_API_KEY", "test-key")

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.raise_for_status.return_value = None

    def fake_which(name):
        return None if name == "rsync" else f"/usr/bin/{name}"

    with patch("capo.remote.lambda_preflight.shutil.which", side_effect=fake_which), \
         patch("capo.remote.lambda_preflight.requests") as fake_requests:
        fake_requests.get.return_value = fake_resp
        result = run_preflight()

    assert result["ok"] is False
    rsync_check = _check_named(result["checks"], "rsync_available")
    assert rsync_check["passed"] is False


def test_preflight_bad_key_permissions(monkeypatch, tmp_path):
    monkeypatch.setenv("LAMBDA_API_KEY", "test-key")

    key = tmp_path / "lambda_main"
    key.write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nbody\n")
    os.chmod(key, 0o644)

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.raise_for_status.return_value = None

    with patch("capo.remote.lambda_preflight.shutil.which", return_value="/usr/bin/x"), \
         patch("capo.remote.lambda_preflight.requests") as fake_requests:
        fake_requests.get.return_value = fake_resp
        result = run_preflight(key_path=str(key))

    assert result["ok"] is False
    perm = _check_named(result["checks"], "ssh_key_permissions")
    assert perm["passed"] is False


def test_preflight_api_auth_fails(monkeypatch):
    monkeypatch.setenv("LAMBDA_API_KEY", "bad-key")

    fake_resp = MagicMock()
    fake_resp.status_code = 401

    with patch("capo.remote.lambda_preflight.shutil.which", return_value="/usr/bin/x"), \
         patch("capo.remote.lambda_preflight.requests") as fake_requests:
        fake_requests.get.return_value = fake_resp
        result = run_preflight()

    assert result["ok"] is False
    auth = _check_named(result["checks"], "api_authenticates")
    assert auth["passed"] is False
    assert "401" in auth["detail"]

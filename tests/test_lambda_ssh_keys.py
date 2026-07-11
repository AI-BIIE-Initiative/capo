"""Tests for find_local_ssh_keys (header-only classification, never reads key
material) and list_remote_ssh_keys (Lambda /ssh-keys API)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from capo.remote.lambda_ssh_keys import find_local_ssh_keys, list_remote_ssh_keys


# ---------------------------------------------------------------------------
# find_local_ssh_keys
# ---------------------------------------------------------------------------

def _write_key(path, header: str, body: str = "SECRETKEYBYTES") -> None:
    path.write_text(f"{header}\n{body}\n-----END PRIVATE KEY-----\n")


def test_find_local_ssh_keys_detects_ed25519(tmp_path):
    key = tmp_path / "lambda_ed25519"
    _write_key(key, "-----BEGIN OPENSSH PRIVATE KEY-----")
    os.chmod(key, 0o600)

    result = find_local_ssh_keys(tmp_path)

    assert len(result) == 1
    entry = result[0]
    assert entry["path"] == str(key)
    assert entry["type"] == "ed25519"  # filename hint refines openssh
    assert entry["permissions_ok"] is True
    # Critical: no key bytes anywhere in the structured output.
    assert "SECRETKEYBYTES" not in str(result)


def test_find_local_ssh_keys_detects_rsa_header(tmp_path):
    key = tmp_path / "id_rsa"
    _write_key(key, "-----BEGIN RSA PRIVATE KEY-----")
    os.chmod(key, 0o600)

    result = find_local_ssh_keys(tmp_path)
    assert len(result) == 1
    assert result[0]["type"] == "rsa"


def test_find_local_ssh_keys_skips_pub_only(tmp_path):
    pub = tmp_path / "id_rsa.pub"
    pub.write_text("ssh-rsa AAAA... user@host\n")

    result = find_local_ssh_keys(tmp_path)
    assert result == []


def test_find_local_ssh_keys_skips_known_hosts_and_config(tmp_path):
    (tmp_path / "known_hosts").write_text("github.com ssh-rsa AAA\n")
    (tmp_path / "config").write_text("Host *\n    User ubuntu\n")
    (tmp_path / "authorized_keys").write_text("ssh-rsa AAA user\n")

    assert find_local_ssh_keys(tmp_path) == []


def test_find_local_ssh_keys_permissions_flag(tmp_path):
    key = tmp_path / "lambda_main"
    _write_key(key, "-----BEGIN OPENSSH PRIVATE KEY-----")
    os.chmod(key, 0o644)

    result = find_local_ssh_keys(tmp_path)
    assert len(result) == 1
    assert result[0]["permissions_ok"] is False


def test_find_local_ssh_keys_unrecognized_header_skipped(tmp_path):
    junk = tmp_path / "id_garbage"
    junk.write_text("not a private key header\n")
    os.chmod(junk, 0o600)

    assert find_local_ssh_keys(tmp_path) == []


def test_find_local_ssh_keys_missing_dir(tmp_path):
    assert find_local_ssh_keys(tmp_path / "does-not-exist") == []


# ---------------------------------------------------------------------------
# list_remote_ssh_keys
# ---------------------------------------------------------------------------

def test_list_remote_ssh_keys_normalises_payload(monkeypatch):
    monkeypatch.setenv("LAMBDA_API_KEY", "test-key")

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "data": [
            {
                "name": "lambda_main",
                "id": "k-1",
                "public_key_fingerprint": "SHA256:abc",
                "extra_field_we_drop": "ignored",
            }
        ]
    }
    fake_resp.raise_for_status.return_value = None

    with patch("capo.remote.lambda_ssh_keys.requests") as fake_requests:
        fake_requests.get.return_value = fake_resp
        result = list_remote_ssh_keys()

    assert len(result) == 1
    assert result[0] == {
        "name": "lambda_main",
        "id": "k-1",
        "public_key_fingerprint": "SHA256:abc",
    }
    # call signature
    fake_requests.get.assert_called_once()
    args, kwargs = fake_requests.get.call_args
    assert args[0].endswith("/api/v1/ssh-keys")
    assert kwargs["auth"] == ("test-key", "")


def test_list_remote_ssh_keys_missing_api_key(monkeypatch):
    monkeypatch.delenv("LAMBDA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="LAMBDA_API_KEY"):
        list_remote_ssh_keys()

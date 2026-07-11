"""Tests for RsyncManager command building — no real shell, no real SSH."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from capo.remote.config import LambdaSessionConfig
from capo.remote.rsync_manager import RsyncManager, RsyncResult


def _make_manager(*, key_path="/home/me/.ssh/lambda_main", excludes=None, remote_workdir="/home/ubuntu/work"):
    config = LambdaSessionConfig(
        host="1.2.3.4",
        user="ubuntu",
        remote_workdir=remote_workdir,
        local_workdir="/local/work",
        key_path=key_path,
        rsync_excludes=excludes if excludes is not None else [".git", "__pycache__"],
    )
    tmux = MagicMock()
    return RsyncManager(config=config, tmux=tmux)


def test_build_rsync_command_uses_private_key():
    mgr = _make_manager(key_path="/home/me/.ssh/lambda_main")
    cmd = mgr._build_rsync_cmd("/local/", "ubuntu@1.2.3.4:/remote/")

    assert cmd[0] == "rsync"
    e_idx = cmd.index("-e")
    e_arg = cmd[e_idx + 1]
    assert "-i /home/me/.ssh/lambda_main" in e_arg
    assert "BatchMode=yes" in e_arg
    assert "StrictHostKeyChecking=no" in e_arg


def test_build_rsync_command_excludes_flatten():
    mgr = _make_manager(excludes=[".git", "__pycache__", "*.pyc"])
    cmd = mgr._build_rsync_cmd("/local/", "ubuntu@1.2.3.4:/remote/")
    assert cmd.count("--exclude") == 3
    assert ".git" in cmd
    assert "__pycache__" in cmd
    assert "*.pyc" in cmd


def test_build_rsync_command_extra_excludes_merged():
    mgr = _make_manager(excludes=[".git"])
    cmd = mgr._build_rsync_cmd(
        "/local/",
        "ubuntu@1.2.3.4:/remote/",
        extra_excludes=["node_modules"],
    )
    assert cmd.count("--exclude") == 2
    assert ".git" in cmd
    assert "node_modules" in cmd


def test_build_rsync_command_no_key_omits_i_flag():
    mgr = _make_manager(key_path=None)
    cmd = mgr._build_rsync_cmd("/local/", "ubuntu@1.2.3.4:/remote/")
    e_arg = cmd[cmd.index("-e") + 1]
    assert " -i " not in e_arg


def test_build_rsync_command_src_dst_at_end():
    mgr = _make_manager()
    cmd = mgr._build_rsync_cmd("/local/", "ubuntu@1.2.3.4:/remote/")
    assert cmd[-2] == "/local/"
    assert cmd[-1] == "ubuntu@1.2.3.4:/remote/"


# ---------------------------------------------------------------------------
# RsyncManager.push integration — patches subprocess.run
# ---------------------------------------------------------------------------

def test_push_invokes_subprocess_with_built_command():
    mgr = _make_manager()
    fake_completed = MagicMock(returncode=0, stdout="ok", stderr="")
    with patch("capo.remote.rsync_manager.subprocess.run", return_value=fake_completed) as fake_run:
        result = mgr.push()
    assert isinstance(result, RsyncResult)
    assert result.success is True
    fake_run.assert_called_once()
    cmd = fake_run.call_args[0][0]
    assert cmd[0] == "rsync"
    # src and dst use the configured workdirs
    assert cmd[-2] == "/local/work/"
    assert cmd[-1] == "ubuntu@1.2.3.4:/home/ubuntu/work/"


def test_push_handles_rsync_failure():
    mgr = _make_manager()
    fake_completed = MagicMock(returncode=23, stdout="", stderr="rsync error")
    with patch("capo.remote.rsync_manager.subprocess.run", return_value=fake_completed):
        result = mgr.push()
    assert result.success is False
    assert "rsync error" in result.stderr


def test_pull_targets_remote_subpath():
    mgr = _make_manager()
    fake_completed = MagicMock(returncode=0, stdout="", stderr="")
    with patch("capo.remote.rsync_manager.subprocess.run", return_value=fake_completed) as fake_run:
        mgr.pull("outputs/", "/local/dest")
    cmd = fake_run.call_args[0][0]
    assert cmd[-2].endswith(":/home/ubuntu/work/outputs/")
    assert cmd[-1] == "/local/dest"
    # pull fetches remote-authoritative artifacts: remote must win, so --update
    # (which would skip files newer on the local side) must NOT be present.
    assert "--update" not in cmd


# ---------------------------------------------------------------------------
# One-directional-safe local → remote sync — never clobber remote run state
# ---------------------------------------------------------------------------

def _exclude_values(cmd):
    return [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "--exclude"]


def test_push_excludes_remote_owned_dirs_and_uses_update():
    mgr = _make_manager()
    fake_completed = MagicMock(returncode=0, stdout="ok", stderr="")
    with patch("capo.remote.rsync_manager.subprocess.run", return_value=fake_completed) as fake_run:
        mgr.push()
    cmd = fake_run.call_args[0][0]
    # --update: a file newer on the remote is never overwritten by a stale local one
    assert "--update" in cmd
    # outputs/, checkpoints/, results/ are remote-owned and must never be pushed up
    excluded = _exclude_values(cmd)
    for owned in ("/outputs", "/checkpoints", "/results"):
        assert owned in excluded, f"{owned} must be excluded from a local→remote push"
    # caller-supplied excludes are still merged in
    assert ".git" in excluded
    # src/dst semantics unchanged (trailing-slash contents copy)
    assert cmd[-2] == "/local/work/"
    assert cmd[-1] == "ubuntu@1.2.3.4:/home/ubuntu/work/"


def test_push_merges_extra_excludes_with_remote_owned():
    mgr = _make_manager()
    fake_completed = MagicMock(returncode=0, stdout="ok", stderr="")
    with patch("capo.remote.rsync_manager.subprocess.run", return_value=fake_completed) as fake_run:
        mgr.push(extra_excludes=["node_modules"])
    excluded = _exclude_values(fake_run.call_args[0][0])
    assert "node_modules" in excluded
    assert "/outputs" in excluded


def test_start_watch_is_one_directional_safe():
    mgr = _make_manager()
    mgr.start_watch("capo-sync:1.0")
    # watch_cmd is the 2nd positional arg to tmux.send_keys(target, cmd, enter=True)
    watch_cmd = mgr.tmux.send_keys.call_args[0][1]
    assert "rsync -avz --update --progress" in watch_cmd
    for owned in ("/outputs", "/checkpoints", "/results"):
        assert f"--exclude '{owned}'" in watch_cmd
    # still a continuous loop pushing local run dir → remote run dir
    assert "while true" in watch_cmd
    assert "/local/work/" in watch_cmd
    assert "ubuntu@1.2.3.4:/home/ubuntu/work/" in watch_cmd


# ---------------------------------------------------------------------------
# Run-scoped workdir guard — bare capo_runs root must be rejected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bad_root",
    [
        "~/capo_runs",
        "~/capo_runs/",
        "~/capo_runs//",
        "/home/ubuntu/capo_runs",
        "/home/ubuntu/capo_runs/",
        "/root/capo_runs",
        "$HOME/capo_runs",
    ],
)
def test_bare_capo_runs_root_is_rejected(bad_root):
    with pytest.raises(ValueError, match="run-scoped"):
        _make_manager(remote_workdir=bad_root)


@pytest.mark.parametrize(
    "good_workdir",
    [
        "~/capo_runs/binding-esm2-20260702-0822-992a",
        "/home/ubuntu/capo_runs/run-001",
        "/home/ubuntu/work",  # unrelated non-capo path stays allowed
        "/local/scratch",
    ],
)
def test_run_scoped_and_unrelated_workdirs_are_allowed(good_workdir):
    mgr = _make_manager(remote_workdir=good_workdir)
    assert mgr.config.remote_workdir == good_workdir


# ---------------------------------------------------------------------------
# upload_run_inputs — the run-upload path used by lambda_upload_run. It must
# ALSO exclude the remote-owned dirs, otherwise the orchestrator's own local
# logs (outputs/run.log) get pushed to the remote and masquerade as the
# training process's logs (this is the binding-boltz-20260703 collision:
# the local run log ended up on the remote at outputs/stdout.log).
# ---------------------------------------------------------------------------

def test_upload_run_inputs_excludes_remote_owned_dirs():
    from pathlib import Path

    from capo.remote import rsync_manager as rm

    fake_completed = MagicMock(returncode=0, stdout="", stderr="")
    with patch.object(rm.subprocess, "run", return_value=fake_completed) as fake_run:
        rm.upload_run_inputs(
            "ubuntu@1.2.3.4",
            Path("/local/capo/binding-boltz-run"),
            "~/capo_runs/binding-boltz-run",
            key_path="/home/me/.ssh/lambda_main",
        )
    # find the rsync call (mkdir calls also go through subprocess.run, but
    # upload_run_inputs itself makes exactly one rsync call)
    rsync_cmds = [c.args[0] for c in fake_run.call_args_list if c.args and c.args[0][0] == "rsync"]
    assert rsync_cmds, "expected an rsync invocation"
    cmd = rsync_cmds[0]
    excluded = _exclude_values(cmd)
    for owned in ("/outputs", "/checkpoints", "/results"):
        assert owned in excluded, f"{owned} must be excluded from the run upload"


def test_upload_run_inputs_merges_caller_excludes_with_remote_owned():
    from pathlib import Path

    from capo.remote import rsync_manager as rm

    fake_completed = MagicMock(returncode=0, stdout="", stderr="")
    with patch.object(rm.subprocess, "run", return_value=fake_completed) as fake_run:
        rm.upload_run_inputs(
            "ubuntu@1.2.3.4",
            Path("/local/capo/run"),
            "~/capo_runs/run",
            excludes=["*.tmp"],
        )
    rsync_cmds = [c.args[0] for c in fake_run.call_args_list if c.args and c.args[0][0] == "rsync"]
    excluded = _exclude_values(rsync_cmds[0])
    assert "*.tmp" in excluded            # caller exclude preserved
    assert "/outputs" in excluded          # remote-owned still enforced


# ---------------------------------------------------------------------------
# Transferred-bytes parser
# ---------------------------------------------------------------------------

def test_parse_transferred_bytes_with_commas():
    stdout = "Total transferred file size: 12,345,678 bytes\n"
    assert RsyncManager._parse_transferred_bytes(stdout) == 12_345_678


def test_parse_transferred_bytes_missing():
    assert RsyncManager._parse_transferred_bytes("nothing relevant\n") == 0

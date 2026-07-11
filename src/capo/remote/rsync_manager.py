from __future__ import annotations

import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from capo.observability import progress as _progress
from capo.remote.config import LambdaSessionConfig
from capo.remote.tmux_manager import TmuxError, TmuxManager


def _check_rsync_installed() -> None:
    if not shutil.which("rsync"):
        raise RuntimeError(
            "rsync is not installed or not on PATH. "
            "Install with:  brew install rsync  (macOS)  or  apt-get install rsync  (Linux)"
        )


_check_rsync_installed()


# The shared capo_runs root, never a valid session workdir. A session whose
# remote_workdir is this bare root makes push() and the background watch loop
# replicate the entire local run tree into ~/capo_runs/ itself, so every run's
# artifacts (checkpoints, outputs, train.py, PIDs, status.json) pile up in one
# shared directory instead of ~/capo_runs/<run_id>/. That collision is what
# confuses the running agent about which PID/status belongs to which run.
# remote_workdir must always be run-scoped: ~/capo_runs/<run_id>.
_CAPO_RUNS_ROOT_RE = re.compile(r"^\s*(~|\$HOME|/home/[^/]+|/root)/capo_runs/*\s*$")


def _assert_run_scoped_workdir(remote_workdir: str) -> None:
    """Reject a remote_workdir that is the shared capo_runs root (no <run_id>).

    Structural backstop for the whole RsyncManager: because both push() and
    start_watch() sync local_workdir/ → remote_workdir/, a bare-root workdir
    dumps the run tree into the shared ~/capo_runs/ directory. Enforced at
    construction so no caller — orchestrator code or an improvising agent —
    can start a session that leaks into the root.
    """
    if _CAPO_RUNS_ROOT_RE.match(remote_workdir or ""):
        raise ValueError(
            f"remote_workdir={remote_workdir!r} is the shared capo_runs root. "
            "A session's remote_workdir must be run-scoped (e.g. "
            "'~/capo_runs/<run_id>') so the one-shot push and the background "
            "rsync watch loop land inside the run folder — never in the shared "
            "root. Pass remote_workdir='~/capo_runs/<run_id>'."
        )


# Run-state directories the REMOTE generates and owns for the entire life of a
# run: training outputs (status.json, metrics.jsonl, train.pid, train.log),
# checkpoints (best/, last/ — the on-instance source of truth, read directly by
# both resume modes), and eval results. Locally these hold only stale
# placeholders, so a local→remote sync must NEVER push them: doing so overwrites
# live run state with an out-of-date copy and confuses the health monitor and the
# running agent about which PID / status / checkpoint is current.
_REMOTE_OWNED_EXCLUDES: tuple[str, ...] = ("/outputs", "/checkpoints", "/results")


@dataclass
class RsyncResult:
    success: bool
    transferred_bytes: int
    duration_s: float
    stdout: str
    stderr: str


class RsyncManager:
    """rsync push/pull/watch for a Lambda session."""

    def __init__(self, config: LambdaSessionConfig, tmux: TmuxManager) -> None:
        _assert_run_scoped_workdir(config.remote_workdir)
        self.config = config
        self.tmux = tmux
        self._watch_pane_target: str | None = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def push(self, extra_excludes: list[str] | None = None) -> RsyncResult:
        """One-shot local → remote sync of INPUTS only.
        One-directional-safe: the remote-owned run-state dirs (outputs/,
        checkpoints/, results/) are always excluded, and --update means a file
        that is newer on the remote (e.g. an in-place edit to a launch script)
        is never overwritten by a stale local copy. A push can therefore only
        ever carry local code/config/data up — it can never clobber live
        training state.
        """
        src = self.config.local_workdir.rstrip("/") + "/"
        dst = (
            f"{self.config.user}@{self.config.host}:"
            f"{self.config.remote_workdir.rstrip('/')}/"
        )
        merged_excludes = list(_REMOTE_OWNED_EXCLUDES) + (extra_excludes or [])
        cmd = self._build_rsync_cmd(
            src, dst, extra_excludes=merged_excludes, extra_flags=["--update"]
        )
        return self._run_rsync(cmd)

    def pull(self, remote_subpath: str, local_dest: str) -> RsyncResult:
        """One-shot remote → local pull (for heavy result artifacts)."""
        remote_base = self.config.remote_workdir.rstrip("/")
        remote_sub = remote_subpath.lstrip("/")
        src = f"{self.config.user}@{self.config.host}:{remote_base}/{remote_sub}"
        dst = local_dest
        cmd = self._build_rsync_cmd(src, dst)
        return self._run_rsync(cmd)

    def start_watch(self, tmux_target: str) -> None:
        """Start a background rsync watch loop in the given tmux pane (non-blocking).

        The loop is one-directional-safe by construction, because it runs for the
        whole session — including while training writes to the remote:
          * it excludes the remote-owned run-state dirs (outputs/, checkpoints/,
            results/) so it can never push a stale local status.json /
            metrics.jsonl / train.pid / checkpoint over the live remote copy;
          * it passes --update so, for the code/config it DOES carry up, a file
            edited more recently on the remote is never overwritten by an older
            local one.
        Its job is strictly to carry local INPUT edits up; remote run state is
        pulled down by the (read-only) monitor and the finalizer, never pushed.
        """
        src = self.config.local_workdir.rstrip("/") + "/"
        dst = (
            f"{self.config.user}@{self.config.host}:"
            f"{self.config.remote_workdir.rstrip('/')}/"
        )
        ssh_e = self._build_ssh_e_flag()
        all_excludes = list(self.config.rsync_excludes) + list(_REMOTE_OWNED_EXCLUDES)
        excludes = " ".join(f"--exclude '{ex}'" for ex in all_excludes)
        interval = self.config.rsync_interval_s

        watch_cmd = (
            f"while true; do "
            f"rsync -avz --update --progress -e '{ssh_e}' {excludes} {src} {dst}; "
            f"echo '[capo-rsync] synced at $(date)'; "
            f"sleep {interval}; "
            f"done"
        )
        self.tmux.send_keys(tmux_target, watch_cmd, enter=True)
        self._watch_pane_target = tmux_target

    def stop_watch(self) -> None:
        """Send Ctrl-C to the watch pane to interrupt the loop."""
        if self._watch_pane_target:
            try:
                self.tmux.send_keys(self._watch_pane_target, "C-c", enter=False)
            except TmuxError:
                pass
            self._watch_pane_target = None

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _build_ssh_e_flag(self) -> str:
        """Build the 'ssh -o ... -i ...' value for rsync's -e argument."""
        parts = ["ssh"]
        for k, v in self.config.ssh_options.items():
            parts += ["-o", f"{k}={v}"]
        if self.config.key_path:
            parts += ["-i", self.config.key_path]
        return " ".join(parts)

    def _build_exclude_flags(self, extra: list[str] | None = None) -> list[str]:
        excludes = list(self.config.rsync_excludes)
        if extra:
            excludes.extend(extra)
        flags: list[str] = []
        for ex in excludes:
            flags += ["--exclude", ex]
        return flags

    def _build_rsync_cmd(
        self,
        src: str,
        dst: str,
        extra_excludes: list[str] | None = None,
        extra_flags: list[str] | None = None,
    ) -> list[str]:
        cmd = [
            "rsync",
            "-avz",
            "--progress",
            "-e", self._build_ssh_e_flag(),
        ]
        if extra_flags:
            cmd += extra_flags
        cmd += self._build_exclude_flags(extra_excludes)
        cmd += [src, dst]
        return cmd

    def _run_rsync(self, cmd: list[str], timeout_s: float = 3600) -> RsyncResult:
        t0 = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            duration = time.monotonic() - t0
            transferred = self._parse_transferred_bytes(result.stdout)
            return RsyncResult(
                success=result.returncode == 0,
                transferred_bytes=transferred,
                duration_s=duration,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        except subprocess.TimeoutExpired:
            return RsyncResult(
                success=False,
                transferred_bytes=0,
                duration_s=time.monotonic() - t0,
                stdout="",
                stderr=f"rsync timed out after {timeout_s}s",
            )

    @staticmethod
    def _parse_transferred_bytes(stdout: str) -> int:
        match = re.search(r"Total transferred file size:\s*([\d,]+)", stdout)
        if match:
            return int(match.group(1).replace(",", ""))
        return 0


# ---------------------------------------------------------------------------
# Standalone run-scoped transfer helpers
# ---------------------------------------------------------------------------

_FORBIDDEN_RSYNC_FLAGS = {"-R", "--relative"}


def _assert_safe_rsync(cmd: list[str]) -> None:
    """Reject rsync commands that would recreate absolute paths on the destination.

    --relative / -R turns /Users/.../runs/<run_id>/foo into
    <dst>/Users/.../runs/<run_id>/foo. Every observed lambda Lambda run
    that leaked a Users/... subtree did so via this flag. Block it at the
    sole subprocess boundary so neither orchestrator code nor an improvising
    agent can put it on the wire.
    """
    bad = _FORBIDDEN_RSYNC_FLAGS.intersection(cmd)
    if bad:
        raise ValueError(
            f"rsync flags {sorted(bad)!r} are forbidden — they recreate the "
            "absolute source path on the destination and produce nested "
            "Users/... directories on the remote. Use trailing-slash semantics "
            "instead (src/ → dst/ copies contents)."
        )


def _rsync(
    src: str,
    dst: str,
    key_path: str | None = None,
    excludes: list[str] | None = None,
    extra_flags: list[str] | None = None,
) -> None:
    """Build and run rsync -avz --partial [--exclude ...] [-e 'ssh -i key'] src dst."""
    cmd = ["rsync", "-avz", "--partial"]
    if excludes:
        for ex in excludes:
            cmd += ["--exclude", ex]
    if key_path:
        cmd += ["-e", f"ssh -i {key_path} -o StrictHostKeyChecking=no"]
    if extra_flags:
        cmd.extend(extra_flags)
    cmd += [src, dst]
    _assert_safe_rsync(cmd)
    subprocess.run(cmd, check=True, capture_output=True)


_REMOTE_RUN_DIR_RE = re.compile(r"^(~|/home/[^/]+)/capo_runs/[A-Za-z0-9._-]+/?$")


def _assert_remote_run_dir_shape(remote_run_dir: str) -> None:
    """Enforce destinations look like ~/capo_runs/<run_id> or /home/<user>/capo_runs/<run_id>.

    Bug A (files leaking to ~/capo_runs/ root, missing <run_id>/ segment)
    came from callers — and from improvising agents — passing a bare or
    truncated remote_run_dir. Validate at the rsync helper boundary so the
    same mistake can never reach the wire from any caller.
    """
    if not _REMOTE_RUN_DIR_RE.match(remote_run_dir):
        raise ValueError(
            f"remote_run_dir={remote_run_dir!r} does not match the canonical "
            "shape '~/capo_runs/<run_id>'. The run_id segment is mandatory — "
            "without it, the rsync would dump files into the shared "
            "~/capo_runs/ root."
        )


def upload_run_inputs(
    ssh_target: str,
    local_run_dir: Path,
    remote_run_dir: str,
    key_path: str | None = None,
    excludes: list[str] | None = None,
) -> None:
    """rsync local_run_dir/ → ssh_target:remote_run_dir/

    The remote-owned run-state dirs (outputs/, checkpoints/, results/) are ALWAYS
    excluded from this push. Without that, the orchestrator's own local logs
    (outputs/run.log, outputs/run_err.log) get uploaded to the remote and
    clobber / masquerade as the training process's logs — which is how a stale
    snapshot of the local run log ended up on the remote at outputs/stdout.log
    and corrupted post-launch diagnosis reads. The remote owns those dirs while
    the run is live; download_run_outputs / sync_run_logs bring them back.
    """
    _assert_remote_run_dir_shape(remote_run_dir)
    _progress.emit(f"[rsync] pushing inputs to {ssh_target}:{remote_run_dir}")
    src = str(local_run_dir).rstrip("/") + "/"
    dst = f"{ssh_target}:{remote_run_dir.rstrip('/')}/"
    merged_excludes = list(_REMOTE_OWNED_EXCLUDES) + list(excludes or [])
    _rsync(src, dst, key_path=key_path, excludes=merged_excludes)
    _progress.emit("[rsync] upload complete")


def push_run_file(
    ssh_target: str,
    local_run_dir: Path,
    run_id: str,
    src_rel: str,
    dst_rel: str | None = None,
    key_path: str | None = None,
) -> None:
    """Push one file from local runs/<run_id>/<src_rel> → ~/capo_runs/<run_id>/<dst_rel>.

    Only the run_id, source-relative-path, and dest-relative-path are accepted —
    callers can never name an absolute remote location. Both src_rel and
    dst_rel are validated to be relative (no leading slash, no .. segments)
    and to land under the canonical run root on both sides.
    """
    dst_rel = dst_rel or src_rel
    for label, rel in (("src_rel", src_rel), ("dst_rel", dst_rel)):
        if rel.startswith("/") or rel.startswith("~"):
            raise ValueError(f"{label}={rel!r} must be relative to the run dir")
        if ".." in Path(rel).parts:
            raise ValueError(f"{label}={rel!r} must not contain '..'")
    local_path = (local_run_dir / src_rel).resolve()
    if not str(local_path).startswith(str(local_run_dir.resolve())):
        raise ValueError(f"src_rel={src_rel!r} resolves outside local_run_dir")
    if not local_path.exists():
        raise FileNotFoundError(f"{local_path} does not exist")
    remote_run_dir = f"~/capo_runs/{run_id}"
    _assert_remote_run_dir_shape(remote_run_dir)
    remote_path = f"{remote_run_dir}/{dst_rel}"
    parent = str(Path(remote_path).parent)
    # Ensure remote parent exists. rsync alone won't create nested dirs.
    mkdir_cmd = ["ssh"]
    if key_path:
        mkdir_cmd += ["-i", key_path, "-o", "StrictHostKeyChecking=no"]
    mkdir_cmd += [ssh_target, f"mkdir -p {parent}"]
    subprocess.run(mkdir_cmd, check=True, capture_output=True)
    _progress.emit(f"[rsync] pushing {src_rel} → {ssh_target}:{remote_path}")
    _rsync(str(local_path), f"{ssh_target}:{remote_path}", key_path=key_path)


def download_run_outputs(
    ssh_target: str,
    remote_run_dir: str,
    local_run_dir: Path,
    key_path: str | None = None,
    subpaths: list[str] | None = None,
) -> None:
    """rsync ssh_target:remote_run_dir/<subpath> → local_run_dir/<subpath> for each subpath.

    Default subpaths bring back everything the remote generated for the run:
    training outputs, eval results, and checkpoints. These are the same dirs
    excluded from local→remote pushes (_REMOTE_OWNED_EXCLUDES) — the remote owns
    them while the run is live, and this is the path that retrieves them once it
    is done. All three always exist on the remote (prepare_remote_run_dir), so a
    default pull never fails on a missing subpath.
    """
    paths = subpaths or ["outputs/", "results/", "checkpoints/"]
    _progress.emit(f"[rsync] downloading results from {ssh_target}")
    for subpath in paths:
        remote_src = f"{ssh_target}:{remote_run_dir.rstrip('/')}/{subpath}"
        local_dst = str(local_run_dir / subpath.rstrip("/")) + "/"
        Path(local_dst).mkdir(parents=True, exist_ok=True)
        _rsync(remote_src, local_dst, key_path=key_path)
    _progress.emit("[rsync] results downloaded")


def sync_run_status(
    ssh_target: str,
    remote_run_dir: str,
    local_run_dir: Path,
    key_path: str | None = None,
) -> None:
    """Sync only the canonical files under outputs/: status.json, metrics.jsonl, train.log, train_err.log."""
    local_run_dir = Path(local_run_dir)
    (local_run_dir / "outputs").mkdir(parents=True, exist_ok=True)
    status_files = [
        "outputs/status.json",
        "outputs/metrics.jsonl",
        "outputs/train.log",
        "outputs/train_err.log",
    ]
    include_flags: list[str] = []
    for f in status_files:
        include_flags += ["--include", f]
    include_flags += ["--exclude", "*"]
    remote_src = f"{ssh_target}:{remote_run_dir.rstrip('/')}/"
    local_dst = str(local_run_dir).rstrip("/") + "/"
    _rsync(remote_src, local_dst, key_path=key_path, extra_flags=include_flags)


def sync_run_logs(
    ssh_target: str,
    remote_run_dir: str,
    local_run_dir: Path,
    key_path: str | None = None,
) -> None:
    """rsync outputs/train.log and outputs/train_err.log only."""
    local_run_dir = Path(local_run_dir)
    (local_run_dir / "outputs").mkdir(parents=True, exist_ok=True)
    log_flags = [
        "--include", "outputs/",
        "--include", "outputs/train.log",
        "--include", "outputs/train_err.log",
        "--exclude", "*",
    ]
    remote_src = f"{ssh_target}:{remote_run_dir.rstrip('/')}/"
    local_dst = str(local_run_dir).rstrip("/") + "/"
    _rsync(remote_src, local_dst, key_path=key_path, extra_flags=log_flags)

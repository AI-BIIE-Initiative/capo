from __future__ import annotations

import shutil
import subprocess
import time

from capo.observability import progress as _progress


def _check_tmux_installed() -> None:
    if not shutil.which("tmux"):
        raise RuntimeError(
            "tmux is not installed or not on PATH. "
            "Install with:  brew install tmux  (macOS)  or  apt-get install tmux  (Linux)"
        )


_check_tmux_installed()


class TmuxError(Exception):
    """Raised when a tmux operation fails or times out."""


class TmuxManager:
    """Low-level tmux session control via subprocess."""

    DEFAULT_CMD_TIMEOUT_S: float = 10.0

    def _run(self, args: list[str], timeout_s: float | None = None) -> str:
        """Run a tmux subcommand. Raises TmuxError on non-zero exit or timeout."""
        cmd = ["tmux"] + args
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s or self.DEFAULT_CMD_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            raise TmuxError(f"tmux command timed out: {' '.join(cmd)}")
        if result.returncode != 0:
            raise TmuxError(
                f"tmux command failed (rc={result.returncode}): {' '.join(cmd)}\n"
                f"stderr: {result.stderr.strip()}"
            )
        return result.stdout.strip()

    def create_session(self, name: str, start_command: str | None = None) -> str:
        """Create a new detached tmux session. Returns the session name."""
        args = ["new-session", "-d", "-s", name]
        if start_command:
            args.append(start_command)
        self._run(args)
        return name

    def has_session(self, name: str) -> bool:
        """Return True if a session with this name already exists."""
        try:
            self._run(["has-session", "-t", name])
            return True
        except TmuxError:
            return False

    def new_window(self, session: str, window_name: str | None = None) -> str:
        """Create a new window in session. Returns 'session:window_index' target."""
        args = [
            "new-window", "-t", session,
            "-P", "-F", "#{session_name}:#{window_index}",
        ]
        if window_name:
            args += ["-n", window_name]
        return self._run(args)

    def send_keys(self, target: str, keys: str, enter: bool = True) -> None:
        """Send keystrokes to a tmux pane target (session:window.pane)."""
        args = ["send-keys", "-t", target, keys]
        if enter:
            args.append("Enter")
        self._run(args)

    def capture_pane(self, target: str, history_lines: int = 500) -> str:
        """Capture visible + scrollback content of a pane as a string."""
        return self._run(
            ["capture-pane", "-pt", target, "-S", f"-{history_lines}"]
        )

    def wait_for_sentinel(
        self,
        target: str,
        sentinel: str,
        timeout_s: float = 300,
        poll_interval: float = 0.5,
    ) -> tuple[bool, str]:
        """
        Poll capture_pane until sentinel string appears or timeout elapses.

        Returns:
            (found, final_output) — found is False on timeout.
        """
        deadline = time.monotonic() + timeout_s
        final_output = ""
        while time.monotonic() < deadline:
            try:
                final_output = self.capture_pane(target)
            except TmuxError:
                pass
            if sentinel in final_output:
                return True, final_output
            time.sleep(poll_interval)
        return False, final_output

    def kill_session(self, name: str) -> None:
        """Kill a tmux session. Silently ignores if it no longer exists."""
        try:
            self._run(["kill-session", "-t", name])
        except TmuxError:
            pass

    def list_sessions(self) -> list[str]:
        """Return names of all active tmux sessions."""
        try:
            output = self._run(["list-sessions", "-F", "#{session_name}"])
            return [s for s in output.splitlines() if s]
        except TmuxError:
            return []


# ---------------------------------------------------------------------------
# Standalone workspace helpers
# ---------------------------------------------------------------------------

_tmux = TmuxManager()


def ensure_local_workspace(
    session_name: str | None = None,
    windows: list[str] | None = None,
) -> None:
    """
    Create local tmux session if it doesn't exist.
    Default windows: [LOCAL_WINDOW_REMOTE, LOCAL_WINDOW_SYNC, LOCAL_WINDOW_LOCAL].
    Idempotent — safe to call if session/windows already exist.
    """
    from capo.remote.config import (
        LOCAL_TMUX_SESSION,
        LOCAL_WINDOW_REMOTE,
        LOCAL_WINDOW_SYNC,
        LOCAL_WINDOW_LOCAL,
    )

    name = session_name or LOCAL_TMUX_SESSION
    win_names = windows or [LOCAL_WINDOW_REMOTE, LOCAL_WINDOW_SYNC, LOCAL_WINDOW_LOCAL]

    if not _tmux.has_session(name):
        _tmux.create_session(name)
        # Rename the auto-created window 0
        try:
            _tmux._run(["rename-window", "-t", f"{name}:0", win_names[0]])
        except TmuxError:
            pass
        for wname in win_names[1:]:
            try:
                _tmux.new_window(name, window_name=wname)
            except TmuxError:
                pass
        _progress.emit(f"[tmux] created local session {name}")
    else:
        # Session exists — ensure each window is present
        for wname in win_names:
            ensure_local_window(name, wname)
        _progress.emit(f"[tmux] local session {name} already active")


def ensure_local_window(
    session_name: str,
    window_name: str,
) -> None:
    """Add window to existing local session if it doesn't already exist."""
    try:
        _tmux.new_window(session_name, window_name=window_name)
    except TmuxError:
        pass  # already exists


def send_to_local_window(
    session_name: str,
    window_name: str,
    command: str,
) -> None:
    """Send a command to a named window in a local tmux session."""
    _tmux.send_keys(target=f"{session_name}:{window_name}", keys=command)


def capture_local_window(
    session_name: str,
    window_name: str,
    lines: int = 200,
) -> str:
    """Capture output from a named window in a local tmux session."""
    return _tmux.capture_pane(target=f"{session_name}:{window_name}", history_lines=lines)


def ensure_remote_tmux(
    ssh_alias: str,
    remote_session_name: str | None = None,
    key_path: str | None = None,
) -> None:
    """
    Ensure capo_remote tmux session exists on the remote host.
    Creates it if it doesn't already exist.
    """
    import subprocess
    from capo.remote.config import REMOTE_TMUX_SESSION

    rsession = remote_session_name or REMOTE_TMUX_SESSION
    cmd = ["ssh"]
    if key_path:
        cmd += ["-i", key_path]
    cmd += [
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        ssh_alias,
        f"tmux has-session -t {rsession} 2>/dev/null || tmux new-session -d -s {rsession}",
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    _progress.emit(f"[tmux] remote session {rsession} ready on {ssh_alias}")


def send_to_remote_tmux(
    ssh_alias: str,
    command: str,
    remote_session_name: str | None = None,
    key_path: str | None = None,
) -> None:
    """Send a command to the remote capo_remote tmux session."""
    import shlex
    import subprocess
    from capo.remote.config import REMOTE_TMUX_SESSION

    rsession = remote_session_name or REMOTE_TMUX_SESSION
    escaped = shlex.quote(command)
    ssh_cmd = ["ssh"]
    if key_path:
        ssh_cmd += ["-i", key_path]
    ssh_cmd += [
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        ssh_alias,
        f"tmux send-keys -t {rsession} {escaped} Enter",
    ]
    subprocess.run(ssh_cmd, check=True, capture_output=True)


def capture_remote_tmux(
    ssh_alias: str,
    remote_session_name: str | None = None,
    lines: int = 200,
    key_path: str | None = None,
) -> str:
    """Capture output from the remote capo_remote tmux session."""
    import subprocess
    from capo.remote.config import REMOTE_TMUX_SESSION

    rsession = remote_session_name or REMOTE_TMUX_SESSION
    ssh_cmd = ["ssh"]
    if key_path:
        ssh_cmd += ["-i", key_path]
    ssh_cmd += [
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        ssh_alias,
        f"tmux capture-pane -t {rsession} -p -S -{lines}",
    ]
    result = subprocess.run(ssh_cmd, capture_output=True, text=True, check=True)
    return result.stdout

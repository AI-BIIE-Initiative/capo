from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Multi-session workspace constants
# ---------------------------------------------------------------------------

LOCAL_TMUX_SESSION: str = "capo"
LOCAL_WINDOW_REMOTE: str = "remote"
LOCAL_WINDOW_SYNC: str = "sync"
LOCAL_WINDOW_LOCAL: str = "local"
REMOTE_TMUX_SESSION: str = "capo_remote"
REMOTE_RUN_ROOT: str = "~/capo_runs"
LOCAL_ARTIFACTS_ROOT: Path = Path.home() / ".capo" / "artifacts"
SSH_READY_TIMEOUT_S: int = 300
SSH_READY_POLL_S: float = 5.0


@dataclass
class LambdaSessionConfig:
    host: str           # raw IP address of the Lambda instance, e.g. "129.153.95.76"
    user: str           # SSH user, always "ubuntu" on Lambda On-Demand
    remote_workdir: str
    local_workdir: str
    key_path: str | None = None
    rsync_excludes: list[str] = field(
        default_factory=lambda: [".git", "__pycache__", "*.pyc", ".DS_Store"]
    )
    rsync_interval_s: int = 30
    ssh_options: dict[str, str] = field(
        default_factory=lambda: {
            "StrictHostKeyChecking": "no",
            "ServerAliveInterval": "30",
            "ServerAliveCountMax": "3",
            "BatchMode": "yes",      # fail fast — never hang waiting for a password prompt
            "ConnectTimeout": "15",
        }
    )
    session_name: str | None = None  # override auto-generated tmux session name

    def ssh_option_flags(self) -> list[str]:
        """Return a flat list of -o Key=Value SSH flags, plus -i key_path if set."""
        flags: list[str] = []
        for k, v in self.ssh_options.items():
            flags += ["-o", f"{k}={v}"]
        if self.key_path:
            flags += ["-i", self.key_path]
        return flags

    def ssh_target(self) -> str:
        """Returns 'user@host', e.g. 'ubuntu@129.153.95.76' — the Lambda SSH login."""
        return f"{self.user}@{self.host}"

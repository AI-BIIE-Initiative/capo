from __future__ import annotations

import json
import os
import secrets
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from capo.mcp.core.types import REPLResult
from capo.mcp.environments.base_env import IsolatedEnv
from capo.observability import progress as _progress
from capo.remote.config import LambdaSessionConfig, SSH_READY_TIMEOUT_S, SSH_READY_POLL_S
from capo.remote.rsync_manager import RsyncManager, RsyncResult
from capo.remote.tmux_manager import TmuxError, TmuxManager


# ---------------------------------------------------------------------------
# Standalone instance-lifecycle helpers
# ---------------------------------------------------------------------------

@dataclass
class LambdaInstance:
    instance_id: str
    ip: str | None
    region: str
    instance_type: str
    status: str              # "booting" | "active" | "unhealthy" | "terminating"
    ssh_key_names: list[str] = field(default_factory=list)
    name: str | None = None
    price_cents_per_hour: int | None = None
    price_dollars_per_hour: float | None = None
    launched_at: str | None = None


def _lambda_api_key(api_key: str | None) -> str:
    key = api_key or os.environ.get("LAMBDA_API_KEY", "")
    if not key:
        raise RuntimeError("LAMBDA_API_KEY environment variable is not set")
    return key


def parse_instance(data: dict[str, Any]) -> LambdaInstance:
    """Parse a Lambda Cloud instance API response into a LambdaInstance.

    Surfaces pricing fields when the payload contains
    `instance_type.price_cents_per_hour`, and the launch timestamp from
    `created_at` / `launched_at` / `launch_time` (whichever is present).
    """
    region_data = data.get("region")
    type_data = data.get("instance_type")

    region = (
        region_data.get("name", "")
        if isinstance(region_data, dict)
        else str(region_data or "")
    )

    if isinstance(type_data, dict):
        instance_type = type_data.get("name", "")
        price_cents = type_data.get("price_cents_per_hour")
    else:
        instance_type = str(type_data or "")
        price_cents = None

    price_dollars = (
        price_cents / 100
        if isinstance(price_cents, (int, float))
        else None
    )

    launched_at = (
        data.get("created_at")
        or data.get("launched_at")
        or data.get("launch_time")
    )

    return LambdaInstance(
        instance_id=data["id"],
        ip=data.get("ip"),
        region=region,
        instance_type=instance_type,
        status=data.get("status", "unknown"),
        ssh_key_names=[
            key["name"] if isinstance(key, dict) else str(key)
            for key in data.get("ssh_key_names", [])
        ],
        name=data.get("name"),
        price_cents_per_hour=price_cents,
        price_dollars_per_hour=price_dollars,
        launched_at=launched_at,
    )


# Legacy private alias retained for any internal caller that survived the rename.
_instance_from_dict = parse_instance


def provision_instance(
    instance_type: str,
    ssh_key_name: str,
    region: str | None = None,
    name: str | None = None,
    api_key: str | None = None,
) -> LambdaInstance:
    """Call Lambda Labs REST API POST /instances. Return LambdaInstance with status=booting."""
    import requests

    _progress.emit(f"Creating Lambda instance {instance_type}{' (' + name + ')' if name else ''}")
    key = _lambda_api_key(api_key)
    payload: dict = {"instance_type_name": instance_type, "ssh_key_names": [ssh_key_name]}
    if region:
        payload["region_name"] = region
    if name:
        payload["name"] = name

    resp = requests.post(
        "https://cloud.lambdalabs.com/api/v1/instance-operations/launch",
        json=payload,
        auth=(key, ""),
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    instance_ids = data.get("data", {}).get("instance_ids", [])
    if not instance_ids:
        raise RuntimeError(f"No instance_ids returned: {data}")
    instance = get_instance(instance_ids[0], api_key=api_key)
    _progress.emit(f"Instance {instance.instance_id} provisioning ({instance.status})")
    return instance


def get_instance(
    instance_id: str,
    api_key: str | None = None,
) -> LambdaInstance:
    """GET /instances/<id>. Return current state."""
    import requests

    key = _lambda_api_key(api_key)
    resp = requests.get(
        f"https://cloud.lambdalabs.com/api/v1/instances/{instance_id}",
        auth=(key, ""),
        timeout=15,
    )
    resp.raise_for_status()
    return parse_instance(resp.json()["data"])


def list_instances(
    api_key: str | None = None,
    *,
    status: str | None = None,
    ssh_key_name: str | None = None,
    region: str | None = None,
    instance_type: str | None = None,
) -> list[LambdaInstance]:
    """GET /instances. Return all instances matching the supplied filters.

    With no filters, returns all visible instances. All provided filters are applied
    as exact-match; omitted filters are ignored. No SSH key filtering occurs unless
    ssh_key_name is explicitly provided.
    """
    import requests

    key = _lambda_api_key(api_key)
    resp = requests.get(
        "https://cloud.lambdalabs.com/api/v1/instances",
        auth=(key, ""),
        timeout=15,
    )
    resp.raise_for_status()
    instances = [parse_instance(d) for d in resp.json().get("data", [])]

    if status is not None:
        instances = [i for i in instances if i.status == status]
    if ssh_key_name is not None:
        instances = [i for i in instances if ssh_key_name in i.ssh_key_names]
    if region is not None:
        instances = [i for i in instances if i.region == region]
    if instance_type is not None:
        instances = [i for i in instances if i.instance_type == instance_type]

    return instances


def wait_for_instance_ip(
    instance_id: str,
    timeout_s: float = SSH_READY_TIMEOUT_S,
    poll_s: float = SSH_READY_POLL_S,
    api_key: str | None = None,
) -> str:
    """Poll get_instance() until ip is non-null. Return ip. Raise TimeoutError."""
    _progress.emit(f"Waiting for instance {instance_id} to become reachable...")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        inst = get_instance(instance_id, api_key=api_key)
        if inst.ip:
            _progress.emit(f"Instance IP assigned: {inst.ip}")
            return inst.ip
        _progress.emit(f"Retrying in {poll_s:.0f}s (instance still booting)...")
        time.sleep(poll_s)
    raise TimeoutError(
        f"Instance {instance_id} did not get an IP within {timeout_s}s"
    )


def wait_for_ssh_ready(
    ip: str,
    key_path: str | Path,
    user: str = "ubuntu",
    timeout_s: float = SSH_READY_TIMEOUT_S,
    poll_s: float = SSH_READY_POLL_S,
) -> None:
    """Try subprocess ssh -o BatchMode=yes ... 'true' until exit code 0. Raise TimeoutError."""
    _progress.emit(f"Waiting for SSH on {user}@{ip}...")
    deadline = time.monotonic() + timeout_s
    cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        "-i", str(key_path),
        f"{user}@{ip}",
        "true",
    ]
    while time.monotonic() < deadline:
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            _progress.emit(f"SSH ready on {user}@{ip}")
            return
        _progress.emit(f"Retrying SSH in {poll_s:.0f}s...")
        time.sleep(poll_s)
    raise TimeoutError(f"SSH to {user}@{ip} not ready within {timeout_s}s")


def ensure_ssh_alias(
    alias: str,
    ip: str,
    key_path: str | Path,
    user: str = "ubuntu",
    ssh_config_path: Path | None = None,
) -> None:
    """
    Write or overwrite a Host block in ~/.ssh/config.
    Idempotent: removes existing block for same alias, then appends fresh block.
    """
    config_path = ssh_config_path or Path.home() / ".ssh" / "config"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    existing = config_path.read_text() if config_path.exists() else ""

    # Remove existing block for this alias
    lines = existing.splitlines(keepends=True)
    new_lines: list[str] = []
    skip = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("host ") and stripped.split()[1] == alias:
            skip = True
            continue
        if skip and stripped.lower().startswith("host "):
            skip = False
        if not skip:
            new_lines.append(line)

    block = (
        f"\nHost {alias}\n"
        f"    HostName {ip}\n"
        f"    User {user}\n"
        f"    IdentityFile {key_path}\n"
        f"    StrictHostKeyChecking no\n"
        f"    ServerAliveInterval 60\n"
    )
    config_path.write_text("".join(new_lines) + block)
    _progress.emit(f"Configured SSH alias {alias} → {ip}")


def terminate_instance(
    instance_id: str,
    api_key: str | None = None,
) -> None:
    """POST /instances/terminate with {"instance_ids": [instance_id]}."""
    import requests

    _progress.emit(f"Terminating instance {instance_id}")
    key = _lambda_api_key(api_key)
    resp = requests.post(
        "https://cloud.lambdalabs.com/api/v1/instance-operations/terminate",
        json={"instance_ids": [instance_id]},
        auth=(key, ""),
        timeout=30,
    )
    resp.raise_for_status()


def safe_terminate_instance(
    instance_id: str,
    expected_ssh_key_names: list[str],
    api_key: str | None = None,
) -> LambdaInstance:
    """Terminate an instance only if its ``ssh_key_names`` contains every name in
    ``expected_ssh_key_names``.

    Raises ``PermissionError`` on mismatch — termination is irreversible, and a
    key mismatch indicates the instance belongs to a different user.
    """
    inst = get_instance(instance_id, api_key=api_key)
    missing = [n for n in expected_ssh_key_names if n not in inst.ssh_key_names]
    if missing:
        raise PermissionError(
            f"Refusing to terminate {instance_id}: instance ssh_key_names="
            f"{inst.ssh_key_names} does not contain expected {missing}."
        )
    terminate_instance(instance_id, api_key=api_key)
    return inst


class LambdaSession(IsolatedEnv):
    """
    Interactive remote session on a Lambda GPU instance.

    Two-window tmux layout created by setup():
      Window 0  "remote"  — SSH interactive shell (agent sends commands here)
      Window 1  "sync"    — Background rsync watch loop (local → remote)

    Command execution uses a sentinel pattern:
      send "<command> ; echo '__CAPO_DONE_<uuid>__'"
      poll capture-pane until sentinel appears
    This guarantees completion detection even for long-running or failing commands.
    """

    SENTINEL_PREFIX = "__CAPO_DONE_"
    SENTINEL_SUFFIX = "__"

    def __init__(
        self,
        config: LambdaSessionConfig,
        tmux: TmuxManager | None = None,
        persistent: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(persistent=persistent, **kwargs)
        self.config = config
        self.tmux = tmux or TmuxManager()
        self.rsync = RsyncManager(config=config, tmux=self.tmux)

        self._token: str = secrets.token_hex(8)
        self._session_id: str = f"lambda_{self._token}"
        self._tmux_session_name: str = (
            config.session_name or f"capo-lambda-{self._token[:8]}"
        )

        self._ssh_pane: str | None = None   # target for window 0, pane 0
        self._sync_pane: str | None = None  # target for window 1, pane 0

        # SupportsPersistence counters
        self._context_count: int = 0
        self._history_count: int = 0
        self._lm_handler_address: tuple[str, int] | None = None

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def tmux_session_name(self) -> str:
        return self._tmux_session_name

    # ------------------------------------------------------------------ #
    #  BaseEnv / IsolatedEnv abstract methods                             #
    # ------------------------------------------------------------------ #

    def setup(self) -> None:
        """
        Create the tmux session with two windows and start the SSH + rsync.

        Raises:
            TmuxError: if the session name already exists.
            RuntimeError: if SSH does not become ready within 30s.
        """
        if self.tmux.has_session(self._tmux_session_name):
            raise TmuxError(
                f"tmux session '{self._tmux_session_name}' already exists. "
                "Use a different session_name or call teardown() first."
            )

        # --- Window 0: remote SSH shell ---
        self.tmux.create_session(self._tmux_session_name)
        self.tmux._run(
            ["rename-window", "-t", f"{self._tmux_session_name}:0", "remote"]
        )
        self._ssh_pane = f"{self._tmux_session_name}:0.0"

        # Build SSH command — mkdir -p ensures remote_workdir exists on first connect
        ssh_flags = self.config.ssh_option_flags()
        remote_cmd = (
            f"mkdir -p {self.config.remote_workdir} && "
            f"cd {self.config.remote_workdir} && "
            f"exec bash"
        )
        ssh_cmd = " ".join(
            ["ssh"] + ssh_flags + ["-t", self.config.ssh_target(), f'"{remote_cmd}"']
        )
        self.tmux.send_keys(self._ssh_pane, ssh_cmd, enter=True)

        # Probe for SSH readiness using a unique sentinel
        ready_sentinel = f"__CAPO_SSH_{self._token[:8]}_READY__"
        time.sleep(5)  # give SSH time to connect before probing
        self.tmux.send_keys(self._ssh_pane, f"echo {ready_sentinel}", enter=True)
        found, _ = self.tmux.wait_for_sentinel(
            self._ssh_pane, ready_sentinel, timeout_s=30
        )
        if not found:
            raise RuntimeError(
                f"SSH connection to {self.config.ssh_target()} did not become ready "
                "within 30s. Check host, key, and network. "
                f"Attach manually with:  tmux attach -t {self._tmux_session_name}"
            )

        # --- Window 1: rsync watch loop ---
        sync_target = self.tmux.new_window(self._tmux_session_name, window_name="sync")
        self._sync_pane = f"{sync_target}.0"
        self.rsync.start_watch(self._sync_pane)

        self.set_session_id(self._session_id)

    def load_context(self, context_payload: dict | list | str) -> None:
        """Inject a context payload as CAPO_CONTEXT_0 in the remote shell."""
        self.add_context(context_payload, context_index=0)

    def execute_code(self, code: str, timeout_s: float = 300) -> REPLResult:
        """
        Execute a shell command on the remote machine.

        For a shell-based remote env, 'code' is treated as a shell command string,
        not Python. Returns a REPLResult for protocol compatibility.
        """
        t0 = time.monotonic()
        found, output = self.execute_raw(code, timeout_s=timeout_s)
        duration = time.monotonic() - t0

        return REPLResult(
            stdout=output,
            stderr=(
                ""
                if found
                else f"Command did not complete within {timeout_s}s (see stdout for partial output)"
            ),
            locals={},
            execution_time=duration,
            llm_calls=[],
        )

    # ------------------------------------------------------------------ #
    #  Core execution                                                      #
    # ------------------------------------------------------------------ #

    def execute_raw(self, command: str, timeout_s: float = 300) -> tuple[bool, str]:
        """
        Send a shell command to the SSH pane and wait for it to complete.

        Uses a per-call unique sentinel to detect completion. The sentinel is
        appended with ';' (not '&&') so it fires regardless of command exit code.

        Returns:
            (found, clean_output) — found is False on timeout.
        """
        if not self._ssh_pane:
            raise RuntimeError("Session not set up. Call setup() first.")

        sentinel_id = uuid.uuid4().hex[:8]
        sentinel = f"{self.SENTINEL_PREFIX}{sentinel_id}{self.SENTINEL_SUFFIX}"

        self.tmux.send_keys(
            self._ssh_pane,
            f"{command} ; echo '{sentinel}'",
            enter=True,
        )

        found, raw_output = self.tmux.wait_for_sentinel(
            self._ssh_pane, sentinel, timeout_s=timeout_s
        )

        # Strip the sentinel line from captured output
        clean_lines = [
            line for line in raw_output.splitlines() if sentinel not in line
        ]
        return found, "\n".join(clean_lines).strip()

    # ------------------------------------------------------------------ #
    #  File sync                                                           #
    # ------------------------------------------------------------------ #

    def sync_push(self) -> RsyncResult:
        """One-shot local → remote sync."""
        return self.rsync.push()

    def sync_pull(self, remote_subpath: str, local_dest: str) -> RsyncResult:
        """One-shot remote → local pull (for large result artifacts)."""
        return self.rsync.pull(remote_subpath, local_dest)

    # ------------------------------------------------------------------ #
    #  Monitoring                                                          #
    # ------------------------------------------------------------------ #

    def get_pane_output(self, lines: int = 200) -> str:
        """Return the last N lines of the SSH pane (useful for monitoring long jobs)."""
        if not self._ssh_pane:
            return ""
        try:
            return self.tmux.capture_pane(self._ssh_pane, history_lines=lines)
        except TmuxError:
            return ""

    def tmux_attach_command(self) -> str:
        """Return the shell command a user can run to attach and watch live."""
        return f"tmux attach -t {self._tmux_session_name}"

    # ------------------------------------------------------------------ #
    #  Teardown                                                            #
    # ------------------------------------------------------------------ #

    def teardown(self) -> None:
        """Stop rsync watch and kill the tmux session (both windows)."""
        self.rsync.stop_watch()
        self.tmux.kill_session(self._tmux_session_name)
        self._ssh_pane = None
        self._sync_pane = None

    # ------------------------------------------------------------------ #
    #  SupportsPersistence protocol                                        #
    # ------------------------------------------------------------------ #

    def update_handler_address(self, address: tuple[str, int]) -> None:
        self._lm_handler_address = address

    def add_context(
        self,
        context_payload: dict | list | str,
        context_index: int | None = None,
    ) -> int:
        if context_index is None:
            context_index = self._context_count

        var_name = f"CAPO_CONTEXT_{context_index}"
        if isinstance(context_payload, str):
            escaped = context_payload.replace("'", "'\\''")
        else:
            escaped = json.dumps(context_payload).replace("'", "'\\''")

        self.execute_raw(f"export {var_name}='{escaped}'")
        self._context_count = max(self._context_count, context_index + 1)
        return context_index

    def get_context_count(self) -> int:
        return self._context_count

    def add_history(
        self,
        message_history: list[dict[str, Any]],
        history_index: int | None = None,
    ) -> int:
        if history_index is None:
            history_index = self._history_count

        remote_path = os.path.join(
            self.config.remote_workdir, f".capo_history_{history_index}.json"
        )
        json_str = json.dumps(message_history).replace("'", "'\\''")
        self.execute_raw(f"echo '{json_str}' > {remote_path}")
        self._history_count = max(self._history_count, history_index + 1)
        return history_index

    def get_history_count(self) -> int:
        return self._history_count

"""
session_store.py — Atomic, lock-protected state file for a fine-tuning run.

Manages `<local_run_dir>/state.json` as the single source of truth for run
phase, config, and recovery hints.

All writes are:
  1. Serialized with an fcntl exclusive lock on a companion .state.lock file
  2. Written to a temp file, fsynced, then os.replace'd atomically

so any concurrent reader sees either the old or the new file, never a partial.
"""

from __future__ import annotations

import errno
import fcntl
import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 1

# Phase progression is monotonic.
# completed and failed are terminal — anything before them is recoverable.
PHASES = (
    "init",         # run_dir created, state.json written, agents not yet dispatched
    "pre_launch",   # pre-launch agent is running
    "training",     # handoff.json present; training is live on Lambda
    "finalizing",   # monitor reached terminal handoff; finalizer is running
    "completed",    # terminal: terminal_state holds the fine-grained outcome
    "failed",       # terminal: orchestrator raised before reaching completion
)

TERMINAL_PHASES = frozenset({"completed", "failed"})


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class SessionState:
    """Complete record of a fine-tuning run — config, layout and phase clock.

    Serialised to <local_run_dir>/state.json at every phase boundary.
    The agent can read this file at any point to understand the run's status
    and how to recover it.
    """

    # --- identity ---
    run_id: str
    created_at: str
    updated_at: str
    schema_version: int = _SCHEMA_VERSION

    # --- run config (mirrors FineTuningOrchestrator constructor args) ---
    model_id: str = ""
    fine_tune_strategy: str = ""
    dataset_ref: str = ""
    ssh_key_name: str = ""
    key_path: str = ""
    gpu_preference: str | None = None
    allow_reuse_existing: bool = True
    max_cost_usd: float = 0.0
    trackio_space_id: str | None = None
    probe_max_retries: int = 3
    ssh_alias_override: str | None = None
    # 3-step gate Step 3 risk appetite (0..1). α = 1 + tolerance_threshold.
    # orchestrator constructor requires it explicitly.
    tolerance_threshold: float = 0.0
    # Reuse a compatible running Lambda instance instead of provisioning a new
    # one (non-interactive default). Mirrors infra.reuse_existing_instance.
    reuse_existing_instance: bool = True
    # The single Lambda instance this run uses (one-instance-per-run invariant).
    # Set after Phase 0 resolves/provisions; a resume must reuse this instance.
    active_instance_id: str | None = None

    # --- layout ---
    local_run_dir: str = ""
    remote_run_dir: str = ""

    # --- phase clock ---
    current_phase: str = "init"
    last_phase_at: str = ""
    terminal_state: str | None = None   # fine-grained outcome when phase is terminal
    error: str | None = None            # set when current_phase == "failed"

    # --- human-readable recovery hint (updated on each phase change) ---
    restart_hint: str = ""

    # --- compaction (stamped by the orchestrator after each context compaction) ---
    compaction_count: int = 0
    last_compaction_at: str | None = None

    # --- paused-for-user-input state (3-step gate user bounce-back) ---
    # Set when the gate hits an info gap only the user can fill (schema, cost overrun).
    # capo_resume.py reads pending_question_path, presents the question, and patches
    # the appropriate artifact before clearing these fields and re-entering the gate.
    paused: bool = False
    pause_reason: str = ""              # e.g. "schema_user_only_info", "cost_accept_overrun"
    pending_question_path: str = ""     # relative to local_run_dir
    pause_context: dict = field(default_factory=dict)


_STATE_FIELD_NAMES = frozenset(f.name for f in fields(SessionState))


def new_session(
    run_id: str,
    local_run_dir: Path | str,
    remote_run_dir: str,
    tolerance_threshold: float,
    model_id: str = "",
    fine_tune_strategy: str = "",
    dataset_ref: str = "",
    ssh_key_name: str = "",
    key_path: str = "",
    gpu_preference: str | None = None,
    allow_reuse_existing: bool = True,
    max_cost_usd: float = 0.0,
    trackio_space_id: str | None = None,
    probe_max_retries: int = 3,
    ssh_alias_override: str | None = None,
    reuse_existing_instance: bool = True,
) -> SessionState:
    """Construct a fresh SessionState for a new run.

    `tolerance_threshold` is required — it must come from the run's YAML
    config, not from a fallback default.
    """
    if not 0.0 <= float(tolerance_threshold) <= 1.0:
        raise ValueError(
            f"tolerance_threshold must be in [0, 1], got {tolerance_threshold!r}"
        )
    now = _now_iso()
    return SessionState(
        run_id=run_id,
        created_at=now,
        updated_at=now,
        last_phase_at=now,
        local_run_dir=str(local_run_dir),
        remote_run_dir=remote_run_dir,
        model_id=model_id,
        fine_tune_strategy=fine_tune_strategy,
        dataset_ref=dataset_ref,
        ssh_key_name=ssh_key_name,
        key_path=key_path,
        gpu_preference=gpu_preference,
        allow_reuse_existing=allow_reuse_existing,
        max_cost_usd=max_cost_usd,
        trackio_space_id=trackio_space_id,
        probe_max_retries=probe_max_retries,
        ssh_alias_override=ssh_alias_override,
        tolerance_threshold=float(tolerance_threshold),
        reuse_existing_instance=reuse_existing_instance,
    )


# ---------------------------------------------------------------------------
# Lock helpers
# ---------------------------------------------------------------------------

@contextmanager
def _flock(lock_path: Path) -> Iterator[None]:
    """Exclusive lock on a companion .lock file; creates it if absent."""
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                time.sleep(0.05)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------

class SessionStore:
    """Atomic, lock-protected state manager for a single fine-tuning run.

    Each instance manages exactly one file: `local_run_dir/state.json`.

    Usage:
        store = SessionStore(local_run_dir)
        store.save(new_session(...))          # initial write
        store.update(current_phase="training") # atomic load-modify-save
        state = store.load()                   # read current state
    """

    def __init__(self, local_run_dir: Path) -> None:
        self._path: Path = local_run_dir / "state.json"
        self._lock_path: Path = local_run_dir / ".state.lock"

    @property
    def path(self) -> Path:
        return self._path

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def save(self, state: SessionState) -> None:
        """Atomically write a new SessionState. Creates state.json."""
        with _flock(self._lock_path):
            self._write_locked(asdict(state))

    def load(self) -> SessionState | None:
        """Return the current SessionState, or None if missing or corrupt."""
        try:
            with _flock(self._lock_path):
                raw = json.loads(self._path.read_text(encoding="utf-8"))
            known = {k: v for k, v in raw.items() if k in _STATE_FIELD_NAMES}
            return SessionState(**known)
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("SessionStore.load failed for %s: %s", self._path, exc)
            return None

    def update(self, **fields_to_set) -> None:
        """Atomic load-modify-save.

        Raises FileNotFoundError if state.json does not exist yet (which
        means save() was never called — a programming error).
        """
        with _flock(self._lock_path):
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            raw.update(fields_to_set)
            raw["updated_at"] = _now_iso()
            if "current_phase" in fields_to_set:
                raw["last_phase_at"] = _now_iso()
            self._write_locked(raw)

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _write_locked(self, data: dict) -> None:
        """Write data atomically. Must be called with the lock held."""
        tmp = self._path.with_suffix(f".{os.getpid()}.{time.time_ns()}.tmp")
        try:
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            with open(tmp, "a") as fd:
                os.fsync(fd.fileno())
            os.replace(tmp, self._path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

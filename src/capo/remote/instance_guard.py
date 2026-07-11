"""Single-Lambda-instance-per-run guard.

A CAPO run may use exactly ONE Lambda instance. It may use an instance with
many GPUs (gpu_8x_h100 is one instance, eight GPUs — fine), but it must
never launch a *second distinct* instance in the same run.

Provisioning runs inside the infra agent's MCP-server process, which is scoped
to a single run, so a process-level record of "the instance this run claimed"
is the right granularity. The guard:

- claim(instance_id) records the run's instance (idempotent for the same id).
- assert_can_provision() raises :class:`SingleInstanceViolation` once an
  instance is already claimed — a second provision would, by definition, create
  a different instance.

The guard auto-resets when the CAPO_RUN_ID environment variable changes, so
a reused process never false-blocks a genuinely new run.
"""

from __future__ import annotations

import os
import threading


class SingleInstanceViolation(RuntimeError):
    """Raised when a run tries to use a second distinct Lambda instance."""


_lock = threading.Lock()
_active_instance_id: str | None = None
_run_token: str | None = None


def _run_id() -> str:
    return os.environ.get("CAPO_RUN_ID", "")


def enforcement_enabled() -> bool:
    """True only inside a real CAPO run (CAPO_RUN_ID set).

    The single-instance invariant is per-run, so standalone/ad-hoc use of the
    provisioning tool (and unit tests of that tool) is not constrained.
    """
    return bool(_run_id())


def _maybe_reset_for_new_run() -> None:
    """Reset state when the ambient run id changes (must hold _lock)."""
    global _run_token, _active_instance_id
    current = _run_id()
    if current != _run_token:
        _run_token = current
        _active_instance_id = None


def reset() -> None:
    """Forget the claimed instance (tests, or an explicit new run)."""
    global _active_instance_id, _run_token
    with _lock:
        _active_instance_id = None
        _run_token = _run_id()


def active_instance_id() -> str | None:
    """The instance id this run has claimed, or None."""
    with _lock:
        _maybe_reset_for_new_run()
        return _active_instance_id


def claim(instance_id: str | None) -> None:
    """Record instance_id as the run's instance.

    Idempotent for the same id (re-attaching the instance the run already uses).
    Claiming a *different* id while one is already claimed is a violation.
    """
    if not instance_id:
        return
    global _active_instance_id
    with _lock:
        _maybe_reset_for_new_run()
        if _active_instance_id is None:
            _active_instance_id = instance_id
        elif _active_instance_id != instance_id:
            raise SingleInstanceViolation(
                "CAPO is limited to one Lambda instance per run. "
                f"Already using instance {_active_instance_id}; "
                f"refusing to also use {instance_id}."
            )


def assert_can_provision() -> None:
    """Raise if the run has already claimed an instance.

    Call this immediately before launching a new Lambda instance. Provisioning
    always creates a new, distinct instance, so once one is claimed any further
    provision is a single-instance violation — reuse the existing one instead.
    """
    with _lock:
        _maybe_reset_for_new_run()
        existing = _active_instance_id
    if existing is not None:
        raise SingleInstanceViolation(
            "CAPO is limited to one Lambda instance per run. "
            f"Already using instance {existing}. Reuse it instead of "
            "provisioning a second instance (multi-GPU is allowed on the same "
            "instance via a larger instance_type)."
        )

"""
post_launch_repair.py — Deterministic, bounded auto-repair for post-launch failures.

After the monitor returns a failure verdict and post_launch_diagnostics classifies
it, the orchestrator may attempt a bounded, *deterministic* repair before giving up
and finalizing. This module owns the remote-side mechanics; the orchestrator owns the
loop, the attempt cap, and the ledger.

Scope is deliberately narrow — only the two categories whose fix is mechanical and
needs no judgement (so a wrong guess can't make things worse):

  missing_dependency  → pip install the named packages, then relaunch.
  cuda_kernel         → install the cuequivariance kernel set; if the RUNTIME import
                        still fails, fall back to --no_kernels (slower pure-PyTorch
                        path, but guaranteed progress); then relaunch.

Categories that need a hyperparameter judgement (oom, nan_inf) or a code patch
(script_bug, data_schema_mismatch) are NOT auto-repaired here — they surface a
precise remediation for a human / the finalizer instead.

Relaunch reuses the run's own idempotent scripts/launch_command.sh, so an
expensive precompute stage (e.g. Boltz embeddings) is reused, not recomputed.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from capo.orchestration.post_launch_diagnostics import _ssh_cmd

# Only these classify as mechanically fixable without judgement.
AUTO_REPAIRABLE: frozenset[str] = frozenset({"missing_dependency", "cuda_kernel"})

# The proven Boltz cuequivariance kernel set (mirrors skills/model-inference/boltz/
# references/boltz2-cuda.lock). Quoted per-token so version specifiers survive the
# remote shell.
_KERNEL_PACKAGES = (
    "cuequivariance cuequivariance-torch cuequivariance-ops-cu12 "
    "cuequivariance-ops-torch-cu12 'platformdirs>=3.0.0' 'networkx>=3.0'"
)
_NVIDIA_INDEX = "--extra-index-url https://pypi.nvidia.com"

_PIP_TIMEOUT_SEC = 900   # cuequivariance wheels are large
_SHELL_TIMEOUT_SEC = 120


@dataclass
class RepairOutcome:
    applied: bool          # the remediation command(s) ran and succeeded
    relaunched: bool       # training was re-launched and a new pid was captured
    new_pid: int | None
    detail: str
    used_no_kernels: bool = False

    def to_dict(self) -> dict:
        return {
            "applied": self.applied,
            "relaunched": self.relaunched,
            "new_pid": self.new_pid,
            "detail": self.detail,
            "used_no_kernels": self.used_no_kernels,
        }


def is_auto_repairable(failure) -> bool:
    """True when this failure is mechanically fixable without judgement."""
    return bool(
        failure is not None
        and getattr(failure, "recoverable", False)
        and getattr(failure, "failure_category", None) in AUTO_REPAIRABLE
    )


def repair_signature(failure) -> tuple:
    """Ledger key for a failure: identical (category, packages) ⇒ identical fix.

    The orchestrator uses this to refuse to apply the SAME fix twice — if a
    missing_dependency recurs after we already installed those packages, the fix
    did not take and re-trying it would loop forever.
    """
    pkgs = tuple(sorted(getattr(failure, "missing_packages", None) or []))
    return (getattr(failure, "failure_category", None), pkgs)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ssh_run(ssh_alias: str, key_path: str | None, remote_cmd: str,
             timeout: int) -> tuple[int, str, str]:
    """Run a remote command over SSH. Returns (returncode, stdout, stderr)."""
    try:
        proc = subprocess.run(
            _ssh_cmd(ssh_alias, key_path, remote_cmd),
            capture_output=True, text=True, timeout=timeout,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired:
        return 124, "", f"ssh timeout after {timeout}s"
    except OSError as exc:
        return 1, "", str(exc)


_PID_RE = re.compile(r"^\s*(\d+)\s*$")


def _parse_pid(text: str) -> int | None:
    """Last all-digits line of stdout is the relaunched pid."""
    for line in reversed((text or "").splitlines()):
        m = _PID_RE.match(line)
        if m:
            return int(m.group(1))
    return None


def _update_handoff_after_relaunch(local_run_dir: Path, new_pid: int,
                                   used_no_kernels: bool) -> None:
    """Point the monitor at the relaunched process and refresh its clock.

    The re-spawned monitor reads reports/handoff.json. Without this it would watch
    the dead pid and treat the ORIGINAL launch's gpu-active deadline (now in the
    past) as an immediate escalation. We update the pid, re-stamp launched_at, and
    drop expected_gpu_active_by_iso so the monitor falls back to heartbeat timing.
    """
    path = local_run_dir / "reports" / "handoff.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    payload["pid"] = new_pid
    payload["launched_at_iso"] = _now_iso()
    payload["auto_repaired"] = True
    payload["used_no_kernels"] = bool(used_no_kernels)
    payload.pop("expected_gpu_active_by_iso", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _relaunch(ssh_alias: str, key_path: str | None, remote_run_dir: str,
              local_run_dir: Path, used_no_kernels: bool,
              detail_parts: list[str]) -> RepairOutcome:
    """Reset stale remote state and re-run the idempotent launch_command.sh.

    Clearing status.json / error markers mirrors the clean pre-launch condition so
    the fresh monitor does not instantly read the previous run's 'failed' status.
    train.log is appended (history preserved); train_err.log is truncated so the
    next diagnosis reads only the relaunched run's traceback.
    """
    reset_and_launch = (
        f"cd {remote_run_dir} && "
        "pkill -f 'python.*train\\.py' || true; sleep 3; "
        "rm -f outputs/status.json outputs/dataset_load_error.json "
        "outputs/canary_failure.json; "
        ": > outputs/train_err.log; "
        "nohup bash scripts/launch_command.sh >> outputs/train.log 2>&1 & "
        "echo $! > outputs/train.pid; sleep 1; cat outputs/train.pid"
    )
    rc, out, err = _ssh_run(ssh_alias, key_path, reset_and_launch,
                            timeout=_SHELL_TIMEOUT_SEC)
    new_pid = _parse_pid(out)
    if new_pid is None:
        return RepairOutcome(
            applied=True, relaunched=False, new_pid=None,
            detail="; ".join(detail_parts + [f"relaunch produced no pid (rc={rc}): {err[-200:]}"]),
            used_no_kernels=used_no_kernels,
        )
    _update_handoff_after_relaunch(local_run_dir, new_pid, used_no_kernels)
    return RepairOutcome(
        applied=True, relaunched=True, new_pid=new_pid,
        detail="; ".join(detail_parts + [f"relaunched pid={new_pid}"]),
        used_no_kernels=used_no_kernels,
    )


def apply_and_relaunch(*, ssh_alias: str, key_path: str | None,
                       remote_run_dir: str, local_run_dir: Path,
                       failure) -> RepairOutcome:
    """Apply the mechanical remediation for failure, then relaunch training.

    Returns a RepairOutcome. relaunched=False means the orchestrator should stop
    repairing and finalize as failed; relaunched=True means a new training process
    is up and the orchestrator should re-monitor it.
    """
    cat = getattr(failure, "failure_category", None)
    pkgs = [p for p in (getattr(failure, "missing_packages", None) or []) if p]
    detail_parts: list[str] = []

    if cat == "missing_dependency":
        if not pkgs:
            return RepairOutcome(
                applied=False, relaunched=False, new_pid=None,
                detail="missing_dependency with no package names — cannot auto-install",
            )
        pkg_args = " ".join(shlex.quote(p) for p in pkgs)
        install = f"cd {remote_run_dir} && pip install -q {_NVIDIA_INDEX} {pkg_args}"
        rc, _out, err = _ssh_run(ssh_alias, key_path, install, timeout=_PIP_TIMEOUT_SEC)
        if rc != 0:
            return RepairOutcome(
                applied=False, relaunched=False, new_pid=None,
                detail=f"pip install {pkgs} failed (rc={rc}): {err[-200:]}",
            )
        detail_parts.append(f"installed {pkgs}")
        # The relaunch + re-monitor IS the verification; the ledger's signature
        # guard stops a loop if the same packages come back missing.
        return _relaunch(ssh_alias, key_path, remote_run_dir, local_run_dir,
                         used_no_kernels=False, detail_parts=detail_parts)

    if cat == "cuda_kernel":
        install = f"cd {remote_run_dir} && pip install -q {_NVIDIA_INDEX} {_KERNEL_PACKAGES}"
        rc, _out, err = _ssh_run(ssh_alias, key_path, install, timeout=_PIP_TIMEOUT_SEC)
        if rc != 0:
            detail_parts.append(f"kernel pip install rc={rc}: {err[-160:]}")
        # Verify the RUNTIME path — importing cuequivariance_torch alone is not
        # enough; the kernel only loads via cuequivariance_ops_torch + the triangle
        # primitive (the exact path Boltz hits at its first GPU forward).
        verify = (
            f"cd {remote_run_dir} && python - <<'PY'\n"
            "import cuequivariance_torch, cuequivariance_ops_torch\n"
            "from cuequivariance_torch.primitives.triangle import "
            "triangle_multiplicative_update\n"
            "print('KERNELS_OK')\n"
            "PY"
        )
        _rc2, out2, _err2 = _ssh_run(ssh_alias, key_path, verify, timeout=_SHELL_TIMEOUT_SEC)
        used_no_kernels = False
        if "KERNELS_OK" in out2:
            detail_parts.append("cuequivariance runtime kernels verified")
        else:
            # Guaranteed-progress fallback: pure-PyTorch path. Idempotent — only add
            # the flag once, only to the boltz predict invocation.
            sed = (
                f"cd {remote_run_dir} && "
                "grep -q -- '--no_kernels' scripts/launch_command.sh || "
                "sed -i 's/boltz predict /boltz predict --no_kernels /' "
                "scripts/launch_command.sh"
            )
            _ssh_run(ssh_alias, key_path, sed, timeout=_SHELL_TIMEOUT_SEC)
            used_no_kernels = True
            detail_parts.append("kernels still failed → added --no_kernels fallback")
        return _relaunch(ssh_alias, key_path, remote_run_dir, local_run_dir,
                         used_no_kernels=used_no_kernels, detail_parts=detail_parts)

    return RepairOutcome(
        applied=False, relaunched=False, new_pid=None,
        detail=f"category {cat!r} is not auto-repairable",
    )

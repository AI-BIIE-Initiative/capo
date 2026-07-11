"""
post_launch_diagnostics.py — Classify post-launch training failures.

Runs after the Haiku monitor returns MonitorHandoff(kind="failed"|"escalation").
Pulls a small set of artifacts from the remote instance over SSH (no LLM),
inspects them for structured failure markers, and writes
reports/post_launch_failure.json with a precise classification.

This bridges the gap between the read-only monitor (which only knows
"process died") and the finalizer (which needs to know why). With the
classification in hand the orchestrator can either:
  - Drive a bounded auto-repair cycle for mechanically-fixable categories
    (missing_dependency / cuda_kernel) — see post_launch_repair.py, or
  - Surface a precise diagnosis and remediation in RUN_REPORT.md.

Failure categories mirror the probe's taxonomy so a single repair ladder
can serve both phases:
  data_schema_mismatch | data_validation_failed | canary_failed
  | missing_dependency | script_bug | oom | nan_inf | cuda_kernel
  | plotting_bug | hub_fallback_stale_cache | no_crash_detected | unknown

IMPORTANT — the classifier is a keyword heuristic, not an oracle. Two rules keep
it honest so downstream recovery is not routed to the wrong fix:
  1. Categories that describe an in-code Python exception (plotting_bug,
     data_schema_mismatch, generic script_bug) are only assigned when the log
     actually contains a traceback marker. A benign warning that merely mentions
     a library name is NOT a crash.
  2. When no crash signature is found at all, the category is `no_crash_detected`
     (the process is probably alive/slow/hung, or dead behind a stale
     status.json) — never a fabricated crash class. The recovery agent then
     verifies liveness before assuming failure.
The remediations below are GENERIC diagnostic scaffolds ("this failed; commonly
caused by A/B/C; inspect X; diagnose and fix step by step"), 
the recovery agent inspects the real evidence and fixes the actual root cause, 
which can vary by task.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
             "-o", "ConnectTimeout=15"]
_PULL_TIMEOUT_SEC = 30
_TRACEBACK_TAIL_LINES = 200

_KEYERROR_RE = re.compile(r"KeyError:\s*['\"]?([^'\"\)]+)['\"]?")
_OOM_RE = re.compile(
    r"(CUDA out of memory|OutOfMemoryError|torch\.cuda\.OutOfMemoryError"
    r"|RuntimeError: out of memory)",
    re.IGNORECASE,
)
_NAN_INF_RE = re.compile(
    r"(loss is nan|loss is inf|FloatingPointError|nan/inf in loss)",
    re.IGNORECASE,
)
_IMPORT_RE = re.compile(r"(ModuleNotFoundError|ImportError):")
# A crash whose traceback runs through the plotting path (generate_plots /
# plot_eval / src/eval/plots.py / a savefig call / a matplotlib STACK FRAME).
# These are COSMETIC: the eval CSVs + checkpoints are written before plotting, so
# the scientific output is intact. The right remediation is to disable inline
# plotting and let training finish (the finalizer regenerates every PNG from the
# CSVs) — NEVER to lose a run over a plot bug. Checked after the hard
# numerical/env failures (OOM / NaN / kernel / import) so a real training crash
# that merely happens near an eval still classifies correctly.

# DELIBERATELY does NOT match the bare word "matplotlib": that substring appears
# in benign startup lines (e.g. the "Unable to import Axes3D … matplotlib …"
# UserWarning matplotlib prints at import time) which are NOT crashes. Matching
# it mislabeled a live, hung CPU baseline as a plotting_bug and nearly killed a
# checkpoint-less run.
_PLOTTING_RE = re.compile(
    r"(generate_plots\("                 # a call to the plotting entry point
    r"|plot_eval"                        # our plotting module / flag
    r"|[\\/]eval[\\/]plots\.py"          # src/eval/plots.py frame
    r"|[\\/]eval[\\/]figures\.py"        # src/eval/figures.py frame
    r"|\.savefig\("                      # a savefig call
    r'|File "[^"]*matplotlib[^"]*"'      # a matplotlib STACK FRAME, not prose
    r")",
    re.IGNORECASE,
)
# Boltz GPU triangular-multiply kernel failures. Deliberately kernel-SPECIFIC:
# we do NOT match a bare "CUDA error" (that is usually a device-side assert =
# model/data bug, which must stay script_bug) — only signatures that point at
# the cuequivariance kernel chain or a torch<->compiled-ext ABI break, so the
# repair ladder reinstalls the kernel set / falls back to --no_kernels rather
# than chasing a code patch. Checked BEFORE _IMPORT_RE so the common
# "No module named 'cuequivariance_torch'" is cuda_kernel, not script_bug.
_CUDA_KERNEL_RE = re.compile(
    r"(cuequivariance"                  # any cuequivariance* module (torch / ops / ops_torch)
    r"|triangle_multiplicative_update"  # the boltz kernel entry point
    r"|triangular_mult"                 # boltz/model/layers/triangular_mult.py
    r"|nvrtc"                           # NVRTC JIT compile failure
    r"|undefined symbol"               # torch<->compiled-ext ABI mismatch
    r"|lib(?:cu|nvrtc)[\w.]*\.so)",     # CUDA shared-lib load failure (libcudart.so.12, …)
    re.IGNORECASE,
)

# Generic missing Python package (NOT a GPU kernel, that is _CUDA_KERNEL_RE,
# checked first and NOT a missing LOCAL module like src.train.x, which is a code
# bug). These are the data/runtime deps (pyarrow, fastparquet, datasets, rdkit, …)
# whose absence makes a loader observe zero columns and then wrongly cry "missing
# columns". The fix is a pip install, so they must classify as missing_dependency,
# never data_schema_mismatch — otherwise the remediation points at the wrong thing.
_MISSING_DEP_RE = re.compile(
    r"(No module named"
    r"|ModuleNotFoundError"
    r"|Missing optional dependency"
    r"|Unable to find a usable engine"
    r"|required for parquet support"
    r"|Could not import module)",
    re.IGNORECASE,
)
# Captures the quoted package in "No module named 'x'" / "Missing optional
# dependency 'x'". Top-level name only (split on the first dot).
_DEP_NAME_RE = re.compile(
    r"(?:No module named|Missing optional dependency)\s*['\"]([A-Za-z0-9_.\-]+)['\"]"
)
# pandas' parquet-engine error names no module, it names the two engines instead.
_PARQUET_ENGINE_RE = re.compile(
    r"usable engine|required for parquet support", re.IGNORECASE
)


def _extract_missing_packages(text: str) -> list[str]:
    """Best-effort list of missing pip package names from an error blob.

    Top-level names only (foo.bar -> foo), de-duplicated, order-preserving.
    The pandas parquet-engine error names no module, so it implies pyarrow/fastparquet.
    """
    pkgs: list[str] = []
    for m in _DEP_NAME_RE.finditer(text or ""):
        name = m.group(1).split(".")[0]
        if name and name not in pkgs:
            pkgs.append(name)
    if _PARQUET_ENGINE_RE.search(text or ""):
        for p in ("pyarrow", "fastparquet"):
            if p not in pkgs:
                pkgs.append(p)
    return pkgs


@dataclass
class PostLaunchFailure:
    failure_category: str
    summary: str
    remediation: str
    recoverable: bool
    failing_file: str | None = None
    missing_columns: list[str] = field(default_factory=list)
    required_columns: list[str] = field(default_factory=list)
    observed_columns: list[str] = field(default_factory=list)
    missing_packages: list[str] = field(default_factory=list)
    hub_lookup_failed: bool | None = None
    cache_path: str | None = None
    cache_mtime_iso: str | None = None
    traceback_tail: str = ""
    diagnosed_at_iso: str = ""
    dataset_load_error_present: bool = False
    train_err_tail_lines: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ssh_cmd(ssh_alias: str, key_path: str | None, remote_cmd: str) -> list[str]:
    cmd = ["ssh", *_SSH_OPTS]
    if key_path:
        cmd += ["-i", key_path]
    cmd += [ssh_alias, remote_cmd]
    return cmd


def _shell_path_arg(remote_path: str) -> str:
    """Render a remote path as a shell argument that STILL expands a leading
    ``~`` / ``$HOME``.

    shlex.quote single-quotes the whole string, and a POSIX shell does NOT
    expand ``~`` inside single quotes — so ``cat '~/capo_runs/x/train_err.log'
    silently fails (exit 1, swallowed by 2>/dev/null) and the diagnostic
    sees an empty traceback, mis-classifying a recoverable crash as
    "unknown / unrecoverable". remote_run_dir is handed to us as
    ~/capo_runs/<run_id> (see fine_tuning_orchestrator), so this path is hit
    on every post-launch diagnosis.

    Fix: expand a leading ``~/`` or ``$HOME/`` to a double-quoted ``"$HOME/"``
    segment (which the shell DOES expand) and single-quote only the safe
    remainder. Adjacent quoted strings concatenate in POSIX sh:
        "$HOME/"'capo_runs/x/train_err.log'  ->  $HOME/capo_runs/x/train_err.log
    Absolute / relative paths fall back to plain shlex.quote.
    """
    if remote_path in ("~", "$HOME"):
        return '"$HOME"'
    for prefix in ("~/", "$HOME/"):
        if remote_path.startswith(prefix):
            return '"$HOME/"' + shlex.quote(remote_path[len(prefix):])
    return shlex.quote(remote_path)


def _remote_read(ssh_alias: str, key_path: str | None, remote_path: str,
                 tail_lines: int | None = None) -> str:
    """Read a remote file via SSH. Returns empty string on any failure.

    When tail_lines is set, only the last N lines are fetched.
    """
    path_arg = _shell_path_arg(remote_path)
    if tail_lines is not None:
        remote_cmd = f"tail -n {int(tail_lines)} {path_arg} 2>/dev/null"
    else:
        remote_cmd = f"cat {path_arg} 2>/dev/null"
    try:
        proc = subprocess.run(
            _ssh_cmd(ssh_alias, key_path, remote_cmd),
            capture_output=True, text=True, timeout=_PULL_TIMEOUT_SEC,
        )
        return proc.stdout or ""
    except (subprocess.TimeoutExpired, OSError):
        return ""


_TRACEBACK_MARKER_RE = re.compile(
    r"Traceback \(most recent call last\):|Uncaught exception", re.IGNORECASE
)


def _read_first_traceback(
    ssh_alias: str,
    key_path: str | None,
    remote_run_dir: str,
    log_names: tuple[str, ...],
) -> str:
    """Return the tail of the first log that contains a traceback marker.

    Reads each outputs/<log_name> tail in order. If any tail contains a
    Python traceback (or an "Uncaught exception" banner), that tail wins — this
    is the log the classifier can actually act on. If none has a traceback, the
    first non-empty tail is returned for context (better than "" so the
    finalizer still sees the last thing the process printed). Empty string only
    if every candidate is empty/unreadable.
    """
    first_nonempty = ""
    for name in log_names:
        text = _remote_read(
            ssh_alias, key_path,
            f"{remote_run_dir}/outputs/{name}",
            tail_lines=_TRACEBACK_TAIL_LINES,
        )
        if not text.strip():
            continue
        if _TRACEBACK_MARKER_RE.search(text):
            return text
        if not first_nonempty:
            first_nonempty = text
    return first_nonempty


def _last_user_file(tb: str) -> str | None:
    """Pick the failing source file from a traceback.

    Prefers the user's own src/ frame (closest to the bug) over the
    framework file at the bottom of the stack.
    """
    user_file = None
    framework_file = None
    for line in tb.splitlines():
        fm = re.search(r'File "([^"]+\.py)"', line)
        if not fm:
            continue
        path = fm.group(1)
        if "/site-packages/" not in path and "/dist-packages/" not in path:
            user_file = path  # keep updating; the last user frame is closest to the bug
        framework_file = path
    return user_file or framework_file


def _classify_from_traceback(tb: str) -> tuple[str, str | None, str]:
    """Return (category, failing_file, summary) from a log/traceback string.

    Runtime failure signatures without a traceback—like CUDA OOM, NaN/Inf aborts, 
    or missing GPU kernels—are treated as direct evidence. Code-exception categories
    are only assigned when a traceback is present, to avoid mistaking benign logs or 
    warnings for crashes. 
    return `no_crash_detected` if nothing matches and there’s no traceback. : means it's likely slow, hung or stale status—not a code crash.

    """
    if not tb:
        return "unknown", None, "No log tail retrievable from remote"
    if _OOM_RE.search(tb):
        return "oom", None, "CUDA out of memory at runtime"
    if _NAN_INF_RE.search(tb):
        return "nan_inf", None, "NaN/Inf detected in training loss"
    km = _CUDA_KERNEL_RE.search(tb)
    if km:
        return (
            "cuda_kernel",
            None,  # environmental, not a source file to patch
            f"A compiled CUDA kernel/extension failed to load or is ABI-incompatible "
            f"(matched {km.group(0)!r})",
        )
    if _IMPORT_RE.search(tb):
        m = _IMPORT_RE.search(tb)
        return "script_bug", None, f"Import error: {m.group(0) if m else 'unknown'}"

    # Everything below describes an in-code exception → require a real traceback
    # marker. Without one there is no crash to classify.
    if not _TRACEBACK_MARKER_RE.search(tb):
        last_line = tb.strip().splitlines()[-1] if tb.strip() else ""
        return (
            "no_crash_detected",
            None,
            "No crash signature in logs (no traceback / OOM / NaN / kernel / import "
            "error) — the process is likely alive (slow or hung) or dead behind a "
            "stale status.json, not a code crash"
            + (f"; last log line: {last_line[:160]}" if last_line else ""),
        )

    # Plotting crashes are checked before KeyError/generic bugs so harmless savefig/plot 
    # failures are routed to the safe “disable inline plots + relaunch” fix, not an unsafe code patch. 
    # This only applies when the traceback shows a real plotting frame, so loader/data KeyErrors are not affected.

    pm = _PLOTTING_RE.search(tb)
    if pm:
        last_line = tb.strip().splitlines()[-1] if tb.strip() else "unknown error"
        return (
            "plotting_bug",
            _last_user_file(tb),
            f"Crash in the plotting path (matched {pm.group(0)!r}): {last_line[:160]}",
        )
    m = _KEYERROR_RE.search(tb)
    if m:
        return (
            "data_schema_mismatch",
            _last_user_file(tb),
            f"KeyError on column {m.group(1)!r} during dataset processing",
        )
    # Generic script bug: a traceback is present but matched no specific class.
    last_line = tb.strip().splitlines()[-1] if tb.strip() else "unknown error"
    return "script_bug", _last_user_file(tb), f"Uncaught exception: {last_line[:200]}"


def _remediation_for(category: str, *, missing_cols: list[str],
                     observed_cols: list[str],
                     failed_checks: list[str] | None = None,
                     missing_packages: list[str] | None = None,
                     summary: str = "") -> tuple[str, bool]:
    """Return (remediation, recoverable) as a GENERIC diagnostic scaffold.

    These strings are advisory context, not fixed repair recipes. 
    They describe the symptom, common causes, artifacts to inspect, and tell 
    the recovery agent to diagnose against real evidence. We avoid hard-coding 
    task-specific fixes because root causes vary. The agent is responsible for the actual repair.

    """
    # A short, quoted preamble so every remediation is anchored to the concrete
    # error the run produced, per "this error {error} happened, this can be due
    # to …".
    obs = f' The failure surfaced as: "{summary.strip()}".' if summary.strip() else ""
    tail = " Diagnose and fix it step by step, driven by the evidence — not by this hint."

    if category == "data_schema_mismatch":
        cols = (
            f" The loader reported missing columns {missing_cols} against observed "
            f"{observed_cols}."
            if missing_cols else ""
        )
        return (
            f"train.py's dataset schema did not match the data it loaded.{obs}{cols} "
            "This is commonly caused by: (1) a column renamed or absent in the "
            "actual dataset, (2) the wrong sequence/label column configured, "
            "(3) the profiled dataset version differing from the one train.py "
            "loaded, or (4) a loader that imported nothing (a missing backend) and "
            "therefore observed zero columns. Inspect the loader under src/data/ and "
            "the observed-vs-required columns, then reconcile the column mapping or "
            f"the dataset.{tail}",
            True,
        )
    if category == "data_validation_failed":
        checks = failed_checks or []
        return (
            f"The on-instance deep data-validation gate failed (failed_checks={checks})."
            f"{obs} This is commonly caused by: (1) a class with zero positives or "
            "zero negatives (which also breaks any pos_weight computation), (2) "
            "label/sequence misalignment, (3) an empty val or test split, or (4) a "
            "bug in how the validator's own inputs are computed. Inspect "
            "outputs/data_validation.json and outputs/dataset_load_error.json "
            "(mirrored locally), confirm the true cause from the numbers, and fix the "
            f"dataset or the validator inputs.{tail}",
            True,
        )
    if category == "canary_failed":
        checks = failed_checks or []
        return (
            f"The short in-training canary failed (failed_checks={checks}) — the "
            f"hyperparameters did not pass the sanity gate before committing GPU "
            f"budget.{obs} This is commonly caused by: (1) learning rate too high "
            "(loss explodes), (2) broken loss masking / NaN gradients, (3) wrong "
            "label dtype or shape, or (4) a head-init scale mismatch. Inspect "
            "outputs/canary_failure.json and outputs/canary_summary.json, identify "
            "which invariant broke, adjust the responsible hyperparameter, and re-run "
            f"the canary before relaunching.{tail}",
            True,
        )
    if category == "oom":
        return (
            f"CUDA ran out of memory during training.{obs} This is commonly caused "
            "by: (1) per-device batch size too large, (2) sequence length too long, "
            "(3) no gradient checkpointing for the activation memory, or (4) an eval "
            "batch larger than the train batch. Reduce the memory footprint (smaller "
            "batch + more grad-accum to preserve the effective batch, gradient "
            "checkpointing, shorter sequences), then re-run the canary at p99 length "
            f"to confirm it fits before relaunching.{tail}",
            True,
        )
    if category == "nan_inf":
        return (
            f"Training loss became NaN/Inf.{obs} This is commonly caused by: (1) "
            "learning rate too high, (2) exploding gradients with no gradient "
            "clipping, (3) fp16 overflow (bf16 is more robust), or (4) a pathological "
            "label distribution or corrupt input. Lower the LR and/or add gradient "
            "clipping and/or switch precision, re-run the canary, then relaunch."
            f"{tail}",
            True,
        )
    if category == "missing_dependency":
        pkgs = [p for p in (missing_packages or []) if p]
        pkg_str = ", ".join(pkgs) if pkgs else "the package(s) named in the error"
        return (
            f"A required Python package is missing on the instance ({pkg_str}).{obs} "
            "This is an ENVIRONMENT gap, not a data/schema problem — a loader that "
            "cannot import its backend reports zero columns, which looks like a "
            "schema error but is not. Install the missing package(s) on the remote "
            "and verify the import succeeds (if it is a GPU/CUDA wheel it may need "
            "the appropriate extra package index). Then pin it in the run's "
            "requirements.txt so the pre-launch env gate catches recurrence in "
            f"seconds before any GPU spend, and re-run.{tail}",
            True,
        )
    if category == "cuda_kernel":
        return (
            f"A compiled CUDA kernel/extension failed to load or is ABI-incompatible "
            f"with the installed torch.{obs} This is commonly caused by: (1) the "
            "kernel package not being installed, (2) a torch<->extension ABI mismatch "
            "(the extension was built against a different torch — reinstall/rebuild "
            "it against the current one), or (3) a missing CUDA shared library. "
            "Reinstall the kernel/extension against the current torch+CUDA and verify "
            "the RUNTIME import path loads it. If it still will not load, fall back to "
            "the model's pure-PyTorch path (e.g. a no-custom-kernel flag) to guarantee "
            f"progress, and reuse any partial artifacts rather than recomputing.{tail}",
            True,
        )
    if category == "plotting_bug":
        return (
            f"The crash is in the PLOTTING path — a plotting stack frame is present."
            f"{obs} This is COSMETIC: eval CSVs and checkpoints are written BEFORE "
            "plotting, so the scientific output is intact. Do NOT lose the run. "
            "Disable inline plotting and relaunch (resuming from the latest checkpoint "
            "if one exists) so training completes and writes all CSVs; the finalizer "
            "regenerates every figure from those CSVs afterward. This is SAFE and "
            "reversible — never patch plot code mid-run and never stop for user "
            "confirmation over a plot bug. (First confirm the crash truly originates "
            f"in plotting and not in an eval/metric computation upstream of it.){tail}",
            True,
        )
    if category == "script_bug":
        return (
            f"An uncaught exception crashed training (a Python traceback is present)."
            f"{obs} Read the traceback, open the failing file at the cited line, and "
            "apply the minimal targeted patch that addresses the ROOT cause, not the "
            "symptom. If the same class of bug was already fixed at probe time, apply "
            f"the analogous fix here.{tail}",
            True,
        )
    if category == "hub_fallback_stale_cache":
        return (
            f"The HF Hub was unreachable and training fell back to a stale cached "
            f"dataset.{obs} Re-run when the Hub is reachable, or pin the dataset to a "
            f"specific revision so the bytes are deterministic.{tail}",
            True,
        )
    if category == "no_crash_detected":
        return (
            f"No crash signature was found in the logs — no traceback, no OOM/NaN, no "
            f"kernel/import error.{obs} This usually means the process is NOT dead. "
            "It may be: (1) alive but slow in a CPU-bound stage (a scikit-learn "
            "baseline, tokenization, or large-file streaming) that the monitor "
            "false-flagged as idle, (2) genuinely hung/deadlocked, or (3) dead behind "
            "a stale status.json. VERIFY LIVENESS FIRST, before assuming failure: "
            "check the pid, whether /proc/<pid>/stat utime advances over a few "
            "seconds, and whether train.log / metrics.jsonl are still growing. If it "
            "is alive and making forward progress, do NOT kill it — let it continue "
            "and extend the monitor deadline. Only if it is truly stuck (no forward "
            f"progress and no CPU advance) treat it as a hang and act.{tail}",
            True,
        )
    return (
        f"The failure could not be classified from the remote artifacts.{obs} Read "
        "outputs/train.log / train_err.log on the remote (or the synced copy) and "
        f"verify whether the process is alive before deciding on a fix.{tail}",
        True,
    )


def diagnose_post_launch_failure(
    *,
    ssh_alias: str,
    key_path: str | None,
    remote_run_dir: str,
    local_run_dir: Path,
) -> PostLaunchFailure | None:
    """Pull failure artifacts from remote, classify, and persist to reports/.

    Returns None if SSH itself fails (no diagnostics possible). Otherwise
    always returns a PostLaunchFailure — even if the category is "unknown" —
    so the finalizer has a stable artifact to read.
    """
    reports_dir = local_run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. dataset_load_error.json — the structured marker the validator writes.
    err_json_text = _remote_read(
        ssh_alias, key_path,
        f"{remote_run_dir}/outputs/dataset_load_error.json",
    )
    dataset_err: dict = {}
    if err_json_text.strip():
        try:
            dataset_err = json.loads(err_json_text)
        except json.JSONDecodeError:
            dataset_err = {}

    # 1b. data_validation.json: written on deep_check success; pull for the
    # finalizer's "Data integrity" section even when training succeeded.
    validation_json_text = _remote_read(
        ssh_alias, key_path,
        f"{remote_run_dir}/outputs/data_validation.json",
    )
    if validation_json_text.strip():
        (local_run_dir / "outputs").mkdir(parents=True, exist_ok=True)
        (local_run_dir / "outputs" / "data_validation.json").write_text(
            validation_json_text, encoding="utf-8",
        )

    # 1c. canary_failure.json — written on canary block failure (exit 6); pull
    # and treat as the authoritative classification when present. canary_summary.json
    # is the success counterpart — mirror it for the finalizer's RUN_REPORT.md.
    canary_err_text = _remote_read(
        ssh_alias, key_path,
        f"{remote_run_dir}/outputs/canary_failure.json",
    )
    canary_err: dict = {}
    if canary_err_text.strip():
        try:
            canary_err = json.loads(canary_err_text)
        except json.JSONDecodeError:
            canary_err = {}
        (local_run_dir / "outputs").mkdir(parents=True, exist_ok=True)
        (local_run_dir / "outputs" / "canary_failure.json").write_text(
            canary_err_text, encoding="utf-8",
        )
    canary_summary_text = _remote_read(
        ssh_alias, key_path,
        f"{remote_run_dir}/outputs/canary_summary.json",
    )
    if canary_summary_text.strip():
        (local_run_dir / "outputs").mkdir(parents=True, exist_ok=True)
        (local_run_dir / "outputs" / "canary_summary.json").write_text(
            canary_summary_text, encoding="utf-8",
        )

    # 2. Traceback tail — for KeyError / OOM / NaN / script-bug classification.
    #    A crash can surface in any of these logs depending on how train.py wires
    #    its handlers: an uncaught exception logged via `logging.exception(...)`
    #    lands in train.log or the nohup'd stdout.log, NOT train_err.log. Reading
    #    only train_err.log (then a single train.log fallback that STOPS at the
    #    first non-empty file) meant a traceback sitting in stdout.log was never
    #    seen — so a plainly-recoverable script bug got mislabeled
    #    "unknown / recoverable=False" and the recovery loop had nothing to act on.
    #    Scan all candidates and PREFER whichever actually contains a traceback
    #    marker; fall back to the first non-empty tail only for context.
    tb_tail = _read_first_traceback(
        ssh_alias, key_path, remote_run_dir,
        ("train_err.log", "train.log", "stderr.log", "stdout.log"),
    )

    # Mirror the artifacts locally for the finalizer.
    if err_json_text.strip():
        (local_run_dir / "outputs").mkdir(parents=True, exist_ok=True)
        (local_run_dir / "outputs" / "dataset_load_error.json").write_text(
            err_json_text, encoding="utf-8",
        )
    if tb_tail.strip():
        (local_run_dir / "outputs").mkdir(parents=True, exist_ok=True)
        (local_run_dir / "outputs" / "train_err.tail.log").write_text(
            tb_tail, encoding="utf-8",
        )

    # 3. Classify. Order of precedence: canary_failure.json (proves the canary
    #    gate fired) > dataset_load_error.json (structured validator contract) >
    #    traceback parsing.
    if canary_err:
        failed_checks = list(canary_err.get("failed_checks") or [])
        summary = str(
            canary_err.get("error_message")
            or f"Canary failed: {failed_checks}"
        )
        remediation, recoverable = _remediation_for(
            "canary_failed", missing_cols=[], observed_cols=[],
            failed_checks=failed_checks, summary=summary,
        )
        failure = PostLaunchFailure(
            failure_category="canary_failed",
            summary=summary,
            remediation=remediation,
            recoverable=recoverable,
            failing_file="src/train/canary.py",
            traceback_tail=tb_tail[-4000:],
            diagnosed_at_iso=_now_iso(),
            dataset_load_error_present=False,
            train_err_tail_lines=len(tb_tail.splitlines()),
        )
        out_path = reports_dir / "post_launch_failure.json"
        out_path.write_text(
            json.dumps(failure.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return failure

    # Missing-dependency / GPU-kernel signatures OVERRIDE a frequently-mislabeled
    # data_schema_mismatch. A loader that cannot import pyarrow/datasets/the kernel
    # observes ZERO columns and wrongly reports "missing required columns"; the true
    # fix is a pip install, not a schema edit. Detect it from the raw error text and
    # short-circuit before trusting dataset_load_error.json's stored category. Runs
    # for the pure-traceback case too (tb_tail is in the blob), so it is the single
    # place this class is classified — no duplicate logic in _classify_from_traceback.
    _blob = " ".join([
        str(dataset_err.get("error_message", "")),
        str(dataset_err.get("summary", "")),
        err_json_text,
        tb_tail,
    ])
    _override: str | None = None
    if _CUDA_KERNEL_RE.search(_blob):
        _override = "cuda_kernel"
    elif (
        _MISSING_DEP_RE.search(_blob)
        and not _OOM_RE.search(_blob)       # a genuine runtime OOM/NaN must win over a
        and not _NAN_INF_RE.search(_blob)   # stray "No module named" deeper in the log
        and [p for p in _extract_missing_packages(_blob) if p != "src"]
    ):
        _override = "missing_dependency"
    if _override is not None:
        pkgs = [p for p in _extract_missing_packages(_blob) if p != "src"]
        summary = (
            f"Required Python package(s) not installed on the instance: {pkgs}"
            if _override == "missing_dependency"
            else "A compiled CUDA kernel/extension is unavailable or ABI-mismatched"
        )
        remediation, recoverable = _remediation_for(
            _override, missing_cols=[], observed_cols=[], missing_packages=pkgs,
            summary=summary,
        )
        failure = PostLaunchFailure(
            failure_category=_override,
            summary=summary,
            remediation=remediation,
            recoverable=recoverable,
            failing_file=None,
            missing_packages=pkgs,
            traceback_tail=tb_tail[-4000:],
            diagnosed_at_iso=_now_iso(),
            dataset_load_error_present=bool(dataset_err),
            train_err_tail_lines=len(tb_tail.splitlines()),
        )
        out_path = reports_dir / "post_launch_failure.json"
        out_path.write_text(
            json.dumps(failure.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return failure

    if dataset_err:
        category = str(dataset_err.get("failure_category") or "data_schema_mismatch")
        missing = list(dataset_err.get("missing_columns") or [])
        observed = list(dataset_err.get("observed_columns") or [])
        required = list(dataset_err.get("required_columns") or [])
        failed_checks = list(dataset_err.get("failed_checks") or [])
        if category == "data_validation_failed":
            summary = str(
                dataset_err.get("error_message")
                or f"Deep validation failed: {failed_checks}"
            )
        else:
            summary = str(
                dataset_err.get("error_message")
                or f"Dataset schema mismatch (missing {missing})"
            )
        remediation, recoverable = _remediation_for(
            category, missing_cols=missing, observed_cols=observed,
            failed_checks=failed_checks, summary=summary,
        )
        failure = PostLaunchFailure(
            failure_category=category,
            summary=summary,
            remediation=remediation,
            recoverable=recoverable,
            failing_file="src/data/dataset.py",
            missing_columns=missing,
            required_columns=required,
            observed_columns=observed,
            hub_lookup_failed=dataset_err.get("hub_lookup_failed"),
            cache_path=dataset_err.get("cache_path"),
            cache_mtime_iso=dataset_err.get("cache_mtime_iso"),
            traceback_tail=tb_tail[-4000:],
            diagnosed_at_iso=_now_iso(),
            dataset_load_error_present=True,
            train_err_tail_lines=len(tb_tail.splitlines()),
        )
    else:
        category, failing_file, summary = _classify_from_traceback(tb_tail)
        remediation, recoverable = _remediation_for(
            category, missing_cols=[], observed_cols=[], summary=summary,
        )
        failure = PostLaunchFailure(
            failure_category=category,
            summary=summary,
            remediation=remediation,
            recoverable=recoverable,
            failing_file=failing_file,
            traceback_tail=tb_tail[-4000:],
            diagnosed_at_iso=_now_iso(),
            dataset_load_error_present=False,
            train_err_tail_lines=len(tb_tail.splitlines()),
        )

    # Persist for the finalizer.
    out_path = reports_dir / "post_launch_failure.json"
    out_path.write_text(
        json.dumps(failure.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return failure

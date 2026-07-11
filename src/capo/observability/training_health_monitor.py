"""Live health monitor for Lambda fine-tuning runs.

Runs concurrently with (and after) the Sonnet pre-launch phase. Waits for
reports/handoff.json (Sonnet writes it when training is confirmed live), then
polls remote training every minute (first 15 min) or every five minutes
(steady state), via a dedicated Haiku AgentRunner. Also accepts a "/health"
command on stdin while the loop is active and triggers an immediate check.

Each report is appended to <local_run_dir>/reports/health/history.jsonl and
printed to stdout. The loop terminates on state==completed, state==failed, or
a severe-severity report, returning a MonitorHandoff the orchestrator uses to
drive the Sonnet finalizer.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from capo.observability import progress as ip
from capo.orchestration.agent_runner import SUBAGENTS, AgentRunner


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HealthReport:
    ts: str
    state: str                                       # running | completed | failed | stalled | unknown | parse_error
    pid_alive: bool | None = None
    epoch: int | None = None
    step: int | None = None
    metrics: dict = field(default_factory=dict)
    baseline_metrics: dict = field(default_factory=dict)
    trend: str = "unknown"                           # improving | plateau | diverging | unknown
    gpu_util_pct: int | None = None
    gpu_mem_pct: int | None = None
    status_age_sec: int | None = None                # now - status.json updated_at (server-side)
    metrics_rows: int | None = None                  # total rows in metrics.jsonl
    last_stdout_line: str = ""
    alerts: list[str] = field(default_factory=list)
    severity: str = "info"                           # info | warn | severe
    summary: str = ""
    trackio_url: str | None = None
    raw: str = ""


@dataclass
class MonitorHandoff:
    """Terminal signal from the monitor loop to the orchestrator."""
    kind: str                                        # completed | failed | escalation | stopped
    reason: str
    last_report: HealthReport | None
    agent_cost_usd: float | None = None


@dataclass
class HealthMonitorContext:
    run_id: str
    local_run_dir: Path
    key_path: str
    remote_run_dir: str
    handoff_path: Path


# ---------------------------------------------------------------------------
# JSON repair
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_json_with_repair(text: str) -> dict | None:
    """Best-effort parse of a JSON object from free-form model output.

    Three tiers:
    1. json.loads on stripped input
    2. strip markdown fences, retry
    3. slice first '{' to last '}', retry
    Returns None if all three fail.
    """
    if not text:
        return None
    candidates = [text.strip()]
    stripped = _FENCE_RE.sub("", text).strip()
    if stripped != text.strip():
        candidates.append(stripped)
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last > first:
        candidates.append(text[first : last + 1])
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _report_from_dict(d: dict, raw: str) -> HealthReport:
    """Build a HealthReport from a parsed JSON dict, tolerating missing keys."""
    def _as_int(v):
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    return HealthReport(
        ts=str(d.get("ts") or datetime.now(timezone.utc).isoformat(timespec="seconds")),
        state=str(d.get("state") or "unknown"),
        pid_alive=d.get("pid_alive") if isinstance(d.get("pid_alive"), bool) else None,
        epoch=_as_int(d.get("epoch")),
        step=_as_int(d.get("step")),
        metrics=dict(d.get("metrics") or {}),
        baseline_metrics=dict(d.get("baseline_metrics") or {}),
        trend=str(d.get("trend") or "unknown"),
        gpu_util_pct=_as_int(d.get("gpu_util_pct")),
        gpu_mem_pct=_as_int(d.get("gpu_mem_pct")),
        status_age_sec=_as_int(d.get("status_age_sec")),
        metrics_rows=_as_int(d.get("metrics_rows")),
        last_stdout_line=str(d.get("last_stdout_line") or ""),
        alerts=list(d.get("alerts") or []),
        severity=str(d.get("severity") or "info"),
        summary=str(d.get("summary") or ""),
        trackio_url=d.get("trackio_url"),
        raw=raw,
    )


# ---------------------------------------------------------------------------
# TrainingHealthMonitor
# ---------------------------------------------------------------------------

class TrainingHealthMonitor:
    FAST_INTERVAL_SEC: float = 60.0
    SLOW_INTERVAL_SEC: float = 300.0
    FAST_WINDOW_SEC: float = 900.0          # first 15 min on fast cadence
    STALL_ESCALATE_TICKS: int = 4           # 4 consecutive stalled ticks => escalate
    HANDOFF_POLL_SEC: float = 10.0

    GPU_IDLE_UTIL_THRESHOLD: int = 5        # util <= this % counts as "GPU not engaged"
    # Fallback deadline when handoff.json omits expected_gpu_active_by_iso:
    # launched_at + this. Generous enough not to false-positive long CPU-bound
    DEFAULT_GPU_ACTIVE_BUDGET_SEC: float = 5400.0    # 90 min
    HEARTBEAT_TIMEOUT_DEFAULT_SEC: float = 900.0     # status.json staleness ceiling
    CADENCE_GAP_FACTOR: float = 2.5         # log when a tick lands >2.5x late

    # --- False-positive protection for the deadline / soft-severe paths --------
    # The costliest monitor MISTAKE (after billing an idle dead GPU) is killing a
    # run that is actually fine: a long CPU-bound pre-GPU stage (a scikit-learn
    # baseline sweep, tokenization, large-file streaming) legitimately shows 0%
    # GPU and 0 metrics for a long time. We distinguish "idle because dead" from
    # "idle because busy on the CPU" by FORWARD PROGRESS between polls (new stdout
    # line / new metrics row / advancing step). When a run is past its GPU-active
    # deadline but still forward-progressing, we GRANT GRACE and warn instead of
    # escalating; we only escalate once it is past deadline AND not progressing for
    # a few consecutive ticks (or its status.json has gone stale, the real
    # silent-crash signal).
    DEADLINE_GRACE_SEC: float = 1800.0          # +30 min granted per extension while progressing
    DEADLINE_MAX_GRACE_SEC: float = 7200.0      # never extend more than +2h past the original deadline
    DEADLINE_ESCALATE_TICKS: int = 3            # consecutive no-progress ticks past deadline before escalating
    SOFT_SEVERE_ESCALATE_TICKS: int = 3         # consecutive soft-severe ticks (no progress) before escalating
    # Severe alerts that are UNAMBIGUOUS crashes — escalate immediately, no debounce.
    HARD_SEVERE_ALERTS: frozenset[str] = frozenset({
        "nan_or_inf_loss", "exploding_grad", "cuda_oom",
        "process_dead_unexpected", "disk_full",
    })

    def __init__(self, ctx: HealthMonitorContext):
        spec = SUBAGENTS["training-health-monitor"]
        self._runner = AgentRunner(
            model_name="claude-haiku-4-5-20251001",
            system_prompt=spec.prompt,
            allowed_tools=list(spec.tools),
            permission_mode="acceptEdits",
            max_turns=12,
            cwd=str(ctx.local_run_dir),
            emit_cost_per_call=False,
        )
        self._ctx = ctx
        self._lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self._cmd_queue: asyncio.Queue[str] = asyncio.Queue()
        self._last_report: HealthReport | None = None
        self._history_path = ctx.local_run_dir / "reports" / "health" / "history.jsonl"
        self._stall_streak = 0
        self._soft_severe_streak = 0
        self._deadline_grace_used = 0.0
        self._deadline_noprogress_streak = 0
        self._launched_at: float | None = None
        self._last_tick_monotonic: float | None = None
        self._gpu_active_deadline_iso: str | None = None
        self._heartbeat_timeout_sec: float = self.HEARTBEAT_TIMEOUT_DEFAULT_SEC
        self._handoff: dict | None = None
        self._stdin_installed = False
        self._stdin_buffer = b""
        self._agent_cost_usd: float = 0.0

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    async def wait_for_handoff(self) -> dict:
        """Block until Sonnet writes reports/handoff.json. Returns its dict, or {} on stop."""
        while not self._stop.is_set():
            if self._ctx.handoff_path.exists():
                try:
                    return json.loads(self._ctx.handoff_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    ip.emit(f"[health] handoff.json present but unreadable: {exc}; retrying")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.HANDOFF_POLL_SEC)
            except asyncio.TimeoutError:
                continue
        return {}

    async def run_loop(self) -> MonitorHandoff:
        """Run until terminal or escalation. Returns the handoff for the orchestrator."""
        handoff = await self.wait_for_handoff()
        if not handoff:
            return self._finalize_handoff(
                MonitorHandoff(kind="stopped", reason="stopped before handoff", last_report=None)
            )
        self._handoff = handoff
        self._launched_at = time.monotonic()
        self._gpu_active_deadline_iso = handoff.get("expected_gpu_active_by_iso")
        try:
            self._heartbeat_timeout_sec = float(
                handoff.get("heartbeat_timeout_sec")
                or self.HEARTBEAT_TIMEOUT_DEFAULT_SEC
            )
        except (TypeError, ValueError):
            self._heartbeat_timeout_sec = self.HEARTBEAT_TIMEOUT_DEFAULT_SEC
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        self._install_stdin_reader()
        trackio_url = handoff.get("trackio_url") or ""
        ip.emit(
            f"[health] Haiku monitor active — trackio: {trackio_url or 'n/a'}"
            + ("  (type /health for on-demand snapshot)" if self._stdin_installed else "")
        )
        try:
            while not self._stop.is_set():
                interval = self._current_interval()
                cmd = await self._wait_for_tick(interval)
                if self._stop.is_set():
                    break
                self._note_cadence_gap(interval)
                on_demand = (cmd == "/health")
                async with self._lock:
                    report = await self._check_once(on_demand=on_demand)
                self._append_history(report)
                self._print(report, on_demand=on_demand)
                # Classify BEFORE overwriting _last_report so the forward-progress
                # checks in _classify_handoff/_deadline_escalation compare THIS
                # report against the PREVIOUS one. Update _last_report only after,
                # so the next tick's _check_once still passes it as previous_report.
                verdict = self._classify_handoff(report)
                self._last_report = report
                if verdict is not None:
                    return self._finalize_handoff(verdict)
            return self._finalize_handoff(
                MonitorHandoff(
                    kind="stopped",
                    reason="stopped by caller",
                    last_report=self._last_report,
                )
            )
        finally:
            self._uninstall_stdin_reader()

    def _finalize_handoff(self, handoff: MonitorHandoff) -> MonitorHandoff:
        """Attach accumulated cost and emit a single end-of-monitor summary line."""
        handoff.agent_cost_usd = self._agent_cost_usd or None
        if self._agent_cost_usd > 0.0:
            ip.emit(f"[summary] monitor_agent_cost=${self._agent_cost_usd:.4f}")
        return handoff

    async def stop(self) -> None:
        self._stop.set()
        # unblock _wait_for_tick
        try:
            self._cmd_queue.put_nowait("__stop__")
        except asyncio.QueueFull:
            pass

    # ------------------------------------------------------------------ #
    # Tick scheduling                                                      #
    # ------------------------------------------------------------------ #

    def _current_interval(self) -> float:
        if self._launched_at is None:
            return self.SLOW_INTERVAL_SEC
        elapsed = time.monotonic() - self._launched_at
        return self.FAST_INTERVAL_SEC if elapsed < self.FAST_WINDOW_SEC else self.SLOW_INTERVAL_SEC

    async def _wait_for_tick(self, timeout: float) -> str | None:
        """Sleep up to timeout seconds, returning early on a stdin command."""
        try:
            cmd = await asyncio.wait_for(self._cmd_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        if cmd == "__stop__":
            return None
        return cmd

    def _note_cadence_gap(self, interval: float) -> None:
        """Self-heartbeat: warn if this tick landed much later than scheduled.

        A billing-sensitive watchdog that goes silent for an hour (e.g. the host
        process was suspended during context compaction) is unacceptable — log
        the gap so reduced coverage is visible instead of invisible.
        """
        now = time.monotonic()
        prev = self._last_tick_monotonic
        self._last_tick_monotonic = now
        if prev is None:
            return
        gap = now - prev
        if gap > self.CADENCE_GAP_FACTOR * interval:
            ip.emit(
                f"[health] WARNING monitor tick was {gap:.0f}s late "
                f"(expected ~{interval:.0f}s) — watchdog stalled "
                f"(host suspended / compaction?); monitoring coverage reduced."
            )

    # ------------------------------------------------------------------ #
    # Stdin reader (TTY only)                                              #
    # ------------------------------------------------------------------ #

    def _install_stdin_reader(self) -> None:
        if sys.platform == "win32":
            return
        try:
            if not sys.stdin.isatty():
                return
        except (ValueError, OSError):
            return
        try:
            loop = asyncio.get_running_loop()
            loop.add_reader(sys.stdin.fileno(), self._on_stdin)
            self._stdin_installed = True
        except (NotImplementedError, OSError, ValueError):
            # Some environments disallow add_reader on stdin
            self._stdin_installed = False

    def _uninstall_stdin_reader(self) -> None:
        if not self._stdin_installed:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.remove_reader(sys.stdin.fileno())
        except (RuntimeError, OSError, ValueError):
            pass
        finally:
            self._stdin_installed = False

    def _on_stdin(self) -> None:
        try:
            chunk = os.read(sys.stdin.fileno(), 4096)
        except (BlockingIOError, OSError):
            return
        if not chunk:
            return
        self._stdin_buffer += chunk
        while b"\n" in self._stdin_buffer:
            line, _, rest = self._stdin_buffer.partition(b"\n")
            self._stdin_buffer = rest
            text = line.decode("utf-8", errors="replace").strip()
            if text.startswith("/health"):
                try:
                    self._cmd_queue.put_nowait("/health")
                except asyncio.QueueFull:
                    pass

    # ------------------------------------------------------------------ #
    # Haiku invocation                                                     #
    # ------------------------------------------------------------------ #

    async def _check_once(self, on_demand: bool) -> HealthReport:
        prompt = self._build_prompt(on_demand=on_demand)
        try:
            result = await self._runner.generate(prompt=prompt)
        except Exception as exc:
            return HealthReport(
                ts=_now_iso(),
                state="unknown",
                summary=f"health check invocation failed: {exc}",
                alerts=["invocation_failed"],
                severity="warn",
            )
        if result.total_cost_usd is not None:
            self._agent_cost_usd += result.total_cost_usd
        raw = result.answer or ""
        parsed = parse_json_with_repair(raw)
        if parsed is None:
            return HealthReport(
                ts=_now_iso(),
                state="parse_error",
                summary="Haiku returned unparseable response",
                alerts=["parse_error"],
                severity="warn",
                raw=raw,
            )
        report = _report_from_dict(parsed, raw=raw)
        # Attach trackio_url from handoff if the model didn't echo it
        if not report.trackio_url and self._handoff:
            report.trackio_url = self._handoff.get("trackio_url")
        return report

    def _build_prompt(self, on_demand: bool) -> str:
        handoff = self._handoff or {}
        prev = (
            json.dumps(asdict(self._last_report), default=str)
            if self._last_report is not None
            else "null"
        )
        # Startup context lets Haiku tell a legitimate long CPU-bound pre-GPU
        # stage (baseline sweep / tokenization / big-file streaming) apart from a
        # stall: during such a phase 0% GPU + 0 metrics is EXPECTED, and a slowly
        # advancing stdout means the run is healthy, not stalled. The launching
        # agent describes this in handoff.notes; pass it through verbatim.
        startup_notes = str(handoff.get("notes") or "").strip()
        return (
            f"Perform a health check now.\n"
            f"\n"
            f"ssh_alias: {handoff.get('ssh_alias', '')}\n"
            f"key_path: {self._ctx.key_path}\n"
            f"remote_run_dir: {handoff.get('remote_run_dir', self._ctx.remote_run_dir)}\n"
            f"run_id: {self._ctx.run_id}\n"
            f"trackio_url: {handoff.get('trackio_url') or 'null'}\n"
            f"expected_gpu_active_by_iso: {handoff.get('expected_gpu_active_by_iso') or 'null'}\n"
            f"startup_context: {startup_notes or 'none provided'}\n"
            f"on_demand: {str(on_demand).lower()}\n"
            f"previous_report: {prev}\n"
            f"\n"
            f"Return ONLY the JSON object defined in your system prompt. "
            f"No prose, no markdown fences."
        )

    # ------------------------------------------------------------------ #
    # Classification                                                       #
    # ------------------------------------------------------------------ #

    def _gpu_active_deadline(self) -> datetime | None:
        """UTC datetime by which the GPU must show real work, or None.

        Prefers handoff.json's expected_gpu_active_by_iso (strategy-specific,
        written by the launching agent); falls back to
        launched_at_iso + DEFAULT_GPU_ACTIVE_BUDGET_SEC. Any grace already
        granted (because the process was still forward-progressing past the
        deadline) is added on top.
        """
        base: datetime | None = None
        dt = _parse_iso(self._gpu_active_deadline_iso)
        if dt is not None:
            base = dt
        else:
            launched = _parse_iso((self._handoff or {}).get("launched_at_iso"))
            if launched is not None:
                base = launched + timedelta(seconds=self.DEFAULT_GPU_ACTIVE_BUDGET_SEC)
        if base is None:
            return None
        return base + timedelta(seconds=self._deadline_grace_used)

    def _forward_progress(self, report: HealthReport) -> bool:
        """True if the process has visibly done NEW work since the last report.

        Forward progress = a new training step, a new metrics row, or a new
        stdout line vs the previous report. This is how we tell a live-but-busy
        CPU-bound stage (a slow baseline / tokenization) apart from a genuinely
        dead or hung process. The first tick (no previous report) counts as
        progress — we never call a run stuck before we have a basis to compare.
        """
        prev = self._last_report
        if prev is None:
            return True
        if (report.step is not None and prev.step is not None
                and report.step > prev.step):
            return True
        if (report.metrics_rows is not None and prev.metrics_rows is not None
                and report.metrics_rows > prev.metrics_rows):
            return True
        cur_line = (report.last_stdout_line or "").strip()
        if cur_line and cur_line != (prev.last_stdout_line or "").strip():
            return True
        return False

    def _deadline_escalation(self, report: HealthReport) -> MonitorHandoff | None:
        """Escalate when the GPU is idle past its deadline AND the run is not
        making forward progress.

        The billing backstop still exists — a silent startup crash that leaves
        an idle GPU behind a stale status.json is the costliest failure mode and
        MUST be caught. But we no longer kill a run that is demonstrably alive
        and working: a long CPU-bound pre-GPU stage (baseline sweep,
        tokenization, big-file streaming) legitimately shows 0% GPU / 0 metrics
        for a while.

        Conditions to even consider escalating: not terminal, GPU idle, past the
        (grace-adjusted) deadline, and no training metrics yet. Once there:
          - if the process is FORWARD-PROGRESSING and grace remains, extend the
            deadline and warn (do NOT escalate);
          - otherwise count a no-progress tick; escalate only after
            DEADLINE_ESCALATE_TICKS consecutive no-progress ticks, OR immediately
            if status.json has gone stale (the classic dead-behind-frozen-status
            signal, which the debounce must not mask).
        """
        if report.state in ("completed", "failed"):
            return None
        if report.gpu_util_pct is None:
            return None
        if report.gpu_util_pct > self.GPU_IDLE_UTIL_THRESHOLD:
            self._deadline_noprogress_streak = 0
            return None
        deadline = self._gpu_active_deadline()
        if deadline is None:
            return None
        now = datetime.now(timezone.utc)
        if now < deadline:
            return None
        no_metrics = (
            (report.metrics_rows in (None, 0))
            and not report.metrics
            and (report.step in (None, 0))
        )
        if not no_metrics:
            # GPU work has started producing metrics — this backstop no longer applies.
            self._deadline_noprogress_streak = 0
            return None

        # A stale status.json is the classic dead-behind-frozen-status signal and
        # must NEVER be masked by grace, not even on the first tick. We know it is
        # stale only when we have a reading; when status_age is unknown (None) we
        # do not treat that alone as stale (SSH hiccup), and lean on progress.
        status_fresh = (
            report.status_age_sec is None
            or report.status_age_sec <= self._heartbeat_timeout_sec
        )
        progressing = self._forward_progress(report)
        if progressing and status_fresh and self._deadline_grace_used < self.DEADLINE_MAX_GRACE_SEC:
            # Live and working, just past a too-tight deadline. Extend + warn.
            self._deadline_grace_used += self.DEADLINE_GRACE_SEC
            self._deadline_noprogress_streak = 0
            ip.emit(
                f"[health] WARNING GPU idle ({report.gpu_util_pct}%) past the "
                f"GPU-active deadline, but the process is still making forward "
                f"progress (last log: {report.last_stdout_line.strip()[:120]!r}) — "
                f"granting +{int(self.DEADLINE_GRACE_SEC // 60)} min grace "
                f"(total +{int(self._deadline_grace_used // 60)} min) instead of "
                f"escalating a live run. This is normal for a long CPU-bound "
                f"pre-GPU stage (baseline / tokenization / streaming)."
            )
            return None

        # Not progressing (or grace exhausted): count toward escalation. Debounce
        # a couple of ticks WHEN status.json is fresh (it may just be a momentary
        # gap between log lines); escalate immediately on a stale status.json.
        self._deadline_noprogress_streak += 1
        if (
            status_fresh
            and self._deadline_noprogress_streak < self.DEADLINE_ESCALATE_TICKS
            and self._deadline_grace_used < self.DEADLINE_MAX_GRACE_SEC
        ):
            ip.emit(
                f"[health] WARNING GPU idle past deadline with no forward progress "
                f"(tick {self._deadline_noprogress_streak}/"
                f"{self.DEADLINE_ESCALATE_TICKS}); status.json still fresh — "
                f"watching one more cycle before escalating."
            )
            return None

        mins = int((now - deadline).total_seconds() // 60)
        stale = (
            f"; status.json is {report.status_age_sec}s stale"
            if report.status_age_sec is not None
            else ""
        )
        why = (
            "no forward progress across "
            f"{self._deadline_noprogress_streak} consecutive check(s)"
            + (" and status.json has gone stale" if not status_fresh else "")
        )
        return MonitorHandoff(
            kind="escalation",
            reason=(
                f"GPU idle ({report.gpu_util_pct}%) with no training progress "
                f"{mins} min past the GPU-active deadline ({deadline.isoformat()})"
                f"{stale} — {why}. Likely a silent startup crash (missing GPU "
                f"kernel dep, weight-load failure), a dead process behind a stale "
                f"status.json, or a hung CPU stage. Stopping the monitor so the "
                f"recovery/finalizer diagnoses instead of billing an idle GPU."
            ),
            last_report=report,
        )

    def _classify_handoff(self, report: HealthReport) -> MonitorHandoff | None:
        # Deterministic billing-critical backstop FIRST: a job past its
        # expected GPU-active deadline with the GPU still idle and no metrics is
        # almost certainly a silent crash. Escalate regardless of the model's
        # reported state/pid_alive — a stale status.json or a reused PID can
        # otherwise mask the death for hours (the $4.70 lesson).
        deadline_handoff = self._deadline_escalation(report)
        if deadline_handoff is not None:
            return deadline_handoff
        if report.state == "completed":
            return MonitorHandoff(
                kind="completed",
                reason=report.summary or "training completed",
                last_report=report,
            )
        if report.state == "failed":
            return MonitorHandoff(
                kind="failed",
                reason=report.summary or "training failed",
                last_report=report,
            )
        if report.severity == "severe":
            hard = [a for a in report.alerts if a in self.HARD_SEVERE_ALERTS]
            if hard:
                # Unambiguous crash (NaN/Inf, OOM, dead process, disk full,
                # exploding grad) — escalate immediately, no debounce.
                self._soft_severe_streak = 0
                return MonitorHandoff(
                    kind="escalation",
                    reason=report.summary or f"severe alerts: {', '.join(hard)}",
                    last_report=report,
                )
            # SOFT-severe (gpu_cold_no_progress / gpu_idle / stalled): these are
            # exactly what a legitimate long CPU-bound pre-GPU stage looks like.
            # If the process is still making forward progress, it is NOT stuck —
            # downgrade to a warning and reset. Otherwise debounce: escalate only
            # after SOFT_SEVERE_ESCALATE_TICKS consecutive no-progress ticks.
            if self._forward_progress(report):
                self._soft_severe_streak = 0
                ip.emit(
                    f"[health] soft-severe alert(s) {report.alerts} but the process "
                    f"is still making forward progress — treating as a warning, not "
                    f"escalating (last log: {report.last_stdout_line.strip()[:120]!r})."
                )
                return None
            self._soft_severe_streak += 1
            if self._soft_severe_streak >= self.SOFT_SEVERE_ESCALATE_TICKS:
                return MonitorHandoff(
                    kind="escalation",
                    reason=(
                        report.summary
                        or f"soft-severe alerts {report.alerts} with no forward "
                        f"progress for {self._soft_severe_streak} consecutive ticks"
                    ),
                    last_report=report,
                )
            ip.emit(
                f"[health] WARNING soft-severe alert(s) {report.alerts} with no "
                f"forward progress (tick {self._soft_severe_streak}/"
                f"{self.SOFT_SEVERE_ESCALATE_TICKS}) — watching before escalating."
            )
            return None
        else:
            self._soft_severe_streak = 0
        if report.state == "stalled":
            # A non-severe 'stalled' verdict is still debounced, and forward
            # progress (a new stdout line since last poll) clears it outright.
            if self._forward_progress(report):
                self._stall_streak = 0
                return None
            self._stall_streak += 1
            if self._stall_streak >= self.STALL_ESCALATE_TICKS:
                return MonitorHandoff(
                    kind="escalation",
                    reason=f"stalled with no forward progress for "
                    f"{self._stall_streak} consecutive ticks",
                    last_report=report,
                )
        else:
            self._stall_streak = 0
        return None

    # ------------------------------------------------------------------ #
    # Output                                                               #
    # ------------------------------------------------------------------ #

    def _append_history(self, report: HealthReport) -> None:
        try:
            with self._history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(report), default=str) + "\n")
        except OSError as exc:
            ip.emit(f"[health] failed to write history.jsonl: {exc}")

    def _print(self, report: HealthReport, on_demand: bool) -> None:
        if on_demand or report.severity in ("warn", "severe"):
            self._print_block(report, on_demand=on_demand)
        else:
            self._print_compact(report)

    def _print_compact(self, r: HealthReport) -> None:
        short_ts = r.ts.split("T")[-1].split("+")[0][:5] if "T" in r.ts else r.ts
        parts = [f"[health {short_ts}]"]
        if r.epoch is not None and r.step is not None:
            parts.append(f"ep {r.epoch} step {r.step}")
        elif r.epoch is not None:
            parts.append(f"ep {r.epoch}")
        tl = r.metrics.get("train_loss")
        vl = r.metrics.get("val_loss")
        mcc = r.metrics.get("val_mcc") or r.metrics.get("mcc")
        auc = r.metrics.get("val_auc") or r.metrics.get("auc")
        if tl is not None:
            parts.append(f"train {tl}")
        if vl is not None:
            parts.append(f"val {vl}")
        if mcc is not None:
            parts.append(f"mcc {mcc}")
        if auc is not None:
            parts.append(f"auc {auc}")
        if r.gpu_util_pct is not None:
            parts.append(f"gpu {r.gpu_util_pct}%")
        trend_marker = {"improving": "↓", "plateau": "·", "diverging": "↑", "unknown": "?"}.get(r.trend, "")
        if trend_marker:
            parts.append(trend_marker)
        ip.emit(" | ".join(parts))

    def _print_block(self, r: HealthReport, on_demand: bool) -> None:
        prefix = "⚠ " if r.severity == "severe" else ("· " if r.severity == "warn" else "")
        header = "/health on-demand" if on_demand else f"{prefix}health alert"
        lines = [
            "",
            f"─── {header} ───",
            f"ts         : {r.ts}",
            f"state      : {r.state}  (severity: {r.severity})",
            f"pid_alive  : {r.pid_alive}",
        ]
        if r.epoch is not None or r.step is not None:
            lines.append(f"progress   : epoch={r.epoch} step={r.step}")
        if r.metrics:
            metric_str = ", ".join(f"{k}={v}" for k, v in r.metrics.items())
            lines.append(f"metrics    : {metric_str}")
        if r.baseline_metrics:
            baseline_str = ", ".join(f"{k}={v}" for k, v in r.baseline_metrics.items())
            lines.append(f"baseline   : {baseline_str}")
        lines.append(f"trend      : {r.trend}")
        if r.gpu_util_pct is not None or r.gpu_mem_pct is not None:
            lines.append(f"gpu        : util={r.gpu_util_pct}%  mem={r.gpu_mem_pct}%")
        if r.last_stdout_line:
            lines.append(f"last log   : {r.last_stdout_line.strip()[:200]}")
        if r.alerts:
            lines.append(f"alerts     : {', '.join(r.alerts)}")
        if r.summary:
            lines.append(f"summary    : {r.summary}")
        if r.trackio_url:
            lines.append(f"trackio    : {r.trackio_url}")
        lines.append("─" * 40)
        for line in lines:
            ip.emit(line)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_iso(raw: object) -> datetime | None:
    """Parse an ISO-8601 timestamp into a tz-aware UTC datetime, or None."""
    if not raw or not isinstance(raw, str):
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

"""
progress.py — Progress emission for all orchestrated runs.

Every tool call the agent makes is parsed here into clean, prefixed, timestamped
log lines.  All state (phase timers, detected instance info) lives on
ProgressEmitter so the module-level helpers can produce richer output as the
run progresses.

Prefix conventions
------------------
[lambda]       instance lifecycle, SSH, remote shell
[tmux]         local and remote tmux session management
[rsync]        file synchronisation (push / pull)
[setup]        file writes, directory creation, dependency installation
[<activity>]   activity-scoped events: model execution, training steps, status
               polling, phase markers. Tag matches activity_tag on ProgressEmitter
               (e.g. "inference", "fine-tuning").
[results]      output collection and inspection
[inspect]      read-only local file/directory queries
[agent]        sub-agent delegation and tool schema loading
[shell]        generic shell commands that don't fit above
[hardware]     GPU / system info queries
[summary]      final one-liner with all run facts

The activity tag is configurable on ProgressEmitter (default "fine-tuning").
All non-activity tags ([lambda], [rsync], [setup], …) are the same across
all run types.

Log files
---------
The CAPO orchestrator's OWN stdout/stderr (this emitter, plus the driver's
stray output) go to the LOCAL run dir:

    All progress lines go to  outputs/run.log      (RUN_LOG_NAME)
    Error lines go to         outputs/run_err.log  (RUN_ERR_LOG_NAME)

These are deliberately NOT named stdout.log / stderr.log: the REMOTE training
process writes its own outputs/train.log (+ train_err.log), and reusing the
"stdout.log" name for the local agent log caused the two to collide across
rsync (the local agent log was pushed onto the remote and masqueraded as the
training log). Keeping the names disjoint makes local↔remote log flow
unambiguous — see rsync_manager.upload_run_inputs / download_run_outputs.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from contextvars import ContextVar, Token
from datetime import datetime

# Canonical filenames for the orchestrator's OWN logs inside a run's outputs/.
# Distinct from the remote training logs (outputs/train.log, train_err.log) so
# local-agent and remote-process logs never collide across rsync.
RUN_LOG_NAME = "run.log"
RUN_ERR_LOG_NAME = "run_err.log"
from pathlib import Path
from typing import Optional


def _silence_sdk_chatter() -> None:
    """Mute third-party INFO chatter so only CAPO's own lines show.

    claude_agent_sdk logs an INFO line for every query ("Using bundled Claude
    Code CLI: …"); other libs (httpx, hf) are similarly noisy. We raise their
    log levels at import time — progress.py is imported by every entry point
    (the CLI and scripts/run_fine_tuning.py both go through the orchestrator),
    so this is the single source-level suppression for the whole system.
    """
    for name in ("claude_agent_sdk", "httpx", "httpcore", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)


_silence_sdk_chatter()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _format_size(n_bytes: int) -> str:
    """Format a byte count as a human-readable string (B / KB / MB / GB / TB)."""
    if n_bytes < 1024:
        return f"{n_bytes} B"
    n = float(n_bytes)
    for unit in ("KB", "MB", "GB", "TB"):
        n /= 1024.0
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
    return f"{n:.1f} TB"


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _ts() -> str:
    """Return current wall-clock time as HH:MM:SS."""
    return datetime.now().strftime("%H:%M:%S")


_INFERENCE_KEYWORDS = (
    "boltz", "esm", "chai", "ankh", "prottrans",
    "alphafold", "run_boltz", "run_inference",
    "predict", "embed", "structure",
)

_TRAINING_KEYWORDS = (
    "train", "finetune", "fine_tune", "fine-tune",
    "lora", "sft", "trainer",
    "train.py", "probe.py", "eval.py",
)

_ACTIVITY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "inference": _INFERENCE_KEYWORDS,
    "fine-tuning": _TRAINING_KEYWORDS,
}


# ---------------------------------------------------------------------------
# SSH parsing helpers
# ---------------------------------------------------------------------------

def _parse_ssh_target_and_remote(raw: str) -> tuple[str, str]:
    """
    Strip 'ssh' and all SSH flags from a command string.
    Returns (target, remote_cmd).
    target may be 'ubuntu@1.2.3.4' or a config alias like 'lambda-infer-7a42e19a'.
    remote_cmd is the unquoted command string, or "" for no remote command.
    """
    s = re.sub(r"^ssh\s+", "", raw.strip())
    # Remove -o Key=val pairs (with or without quotes)
    s = re.sub(r"-o\s+\S+(?:=\S*)?\s*", "", s)
    # Remove -i /path pairs
    s = re.sub(r"-i\s+\S+\s*", "", s)
    # Remove single-char flags (-T -q -A -n -N etc.)
    s = re.sub(r"-[a-zA-Z]\s+", "", s)
    s = s.strip()

    m = re.match(r"^(\S+)\s*(.*)", s, re.DOTALL)
    if not m:
        return s, ""
    target = m.group(1)
    rest = m.group(2).strip()

    # Strip outermost quotes from remote command
    if len(rest) >= 2:
        if (rest[0] == '"' and rest[-1] == '"') or (rest[0] == "'" and rest[-1] == "'"):
            rest = rest[1:-1].strip()

    return target, rest


def _describe_poll_purpose(cmd: str) -> str:
    """Describe what a polling command is checking."""
    if "status.json" in cmd:
        return "Job still running; polling status"
    if "outputs" in cmd or (re.search(r"\bls\b", cmd) and "capo_runs" in cmd):
        return "Job still running; checking for output files"
    if re.search(r"\becho\b|\btrue\b|\bwhoami\b", cmd):
        return "Probing SSH connectivity"
    return "Waiting before next remote check"


# ---------------------------------------------------------------------------
# ProgressEmitter
# ---------------------------------------------------------------------------

class ProgressEmitter:
    """
    Write timestamped progress lines to the terminal and to log files.

    Stateful: tracks phase timers and detected instance metadata so that
    successive tool calls can produce progressively richer messages
    (e.g. instance-type label, activity start/end timing).
    """

    def __init__(
        self,
        stdout_log: Optional[Path] = None,
        stderr_log: Optional[Path] = None,
        activity_tag: str = "inference",
        console: bool | None = None,
    ) -> None:
        self._stdout_log = stdout_log
        self._stderr_log = stderr_log
        self._activity_tag = activity_tag
        # When False, lines are still written to the log files but NOT printed to
        # the terminal. The capo CLI sets CAPO_PROGRESS_CONSOLE=0 so its Rich
        # log streamer (which tails the log files) is the single terminal source
        
        # avoids double-printing under prompt_toolkit's patch_stdout. Default
        # (env unset) keeps the original behaviour: print to the terminal.
        if console is None:
            console = os.environ.get("CAPO_PROGRESS_CONSOLE", "1") != "0"
        self._console = console

        # Phase timing state
        self._phase_timers: dict[str, float] = {}

        # Detected run state
        self._instance_ip: str = ""
        self._instance_type: str = ""
        self._instance_reused: Optional[bool] = None   # True=reuse, False=new, None=unknown
        self._potential_reuse_ip: str = ""             # IP found in list; cleared on launch
        self._activity_started: bool = False
        self._activity_t0: float = 0.0

        # Billing window tracking
        self._billing_start_t: Optional[float] = None
        self._billing_start_ts: str = ""
        self._results_end_t: Optional[float] = None
        self._results_end_ts: str = ""
        self._activity_start_ts: str = ""
        self._activity_end_ts: str = ""
        self._ssh_target: str = ""  # alias or IP (whatever the agent uses)

        # Last tool call (for result correlation)
        self._last_tool_name: str = ""
        self._last_bash_cmd: str = ""

        if stdout_log:
            stdout_log.parent.mkdir(parents=True, exist_ok=True)
            stdout_log.touch(exist_ok=True)
        if stderr_log:
            stderr_log.parent.mkdir(parents=True, exist_ok=True)
            stderr_log.touch(exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public emit API                                                       #
    # ------------------------------------------------------------------ #

    def emit(self, message: str) -> None:
        """Print one progress line to stdout and append to stdout_log."""
        tag = _runner_tag.get()
        body = f"[{tag}] {message}" if tag else message
        line = f"{_ts()} {body}"
        if self._console:
            print(line, flush=True)
        if self._stdout_log:
            with self._stdout_log.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def error(self, message: str) -> None:
        """Print one error line to stderr and append to stderr_log."""
        tag = _runner_tag.get()
        body = f"[{tag}] {message}" if tag else message
        line = f"{_ts()} {body}"
        if self._console:
            print(line, file=sys.stderr, flush=True)
        if self._stderr_log:
            with self._stderr_log.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def emit_tool_call(self, tool_name: str, tool_input: dict) -> None:
        """Format and emit all progress lines for a single tool call."""
        self._last_tool_name = tool_name
        if tool_name == "Bash":
            self._last_bash_cmd = tool_input.get("command", "")
        for line in self._format_tool_call(tool_name, tool_input):
            self.emit(line)

    def emit_tool_result(self, tool_name: str, result_text: str) -> None:
        """
        Parse a tool result and emit any meaningful follow-up lines.
        Called from agent_runner after each tool execution.
        """
        for line in self._parse_tool_result(tool_name, result_text):
            self.emit(line)

    def phase_start(self, name: str) -> None:
        self._phase_timers[name] = time.monotonic()
        self.emit(f"[{name}] started_at={_ts()}")

    def phase_end(self, name: str) -> None:
        t0 = self._phase_timers.pop(name, None)
        dur = _format_duration(time.monotonic() - t0) if t0 is not None else "?"
        self.emit(f"[{name}] finished_at={_ts()} duration={dur}")

    def mark_results_ready(self) -> None:
        """
        Call from orchestration.py after output files are confirmed locally.
        Records end-of-billing timestamp and emits a results-ready line.
        """
        self._results_end_t = time.monotonic()
        self._results_end_ts = _ts()
        if self._activity_started:
            self._activity_started = False
            if not self._activity_end_ts:
                self._activity_end_ts = self._results_end_ts
        self.emit("[results] All results downloaded locally")

    def emit_final_summary(
        self,
        run_id: str,
        state: str,
        agent_cost_usd: Optional[float],
    ) -> None:
        """
        Emit timing [summary] lines at the end of a run.
        Infrastructure cost is calculated and reported by the agent itself
        (see prompt step §13 in orchestration.py).
        """
        # Determine billing window for elapsed time
        billing_start_t = self._billing_start_t
        billing_start_ts = self._billing_start_ts
        if self._instance_reused and self._activity_t0:
            billing_start_t = self._activity_t0
            billing_start_ts = self._activity_start_ts

        if billing_start_t is not None:
            end_t = self._results_end_t or time.monotonic()
            end_ts = self._results_end_ts or _ts()
            dur_s = end_t - billing_start_t
            self.emit(
                f"[billing_window] start={billing_start_ts} end={end_ts} "
                f"duration={_format_duration(dur_s)}"
            )

        # Build summary line
        if self._instance_reused is True:
            instance_part = f"reused:{self._ssh_target or self._instance_ip}"
        elif self._instance_reused is False:
            instance_part = (
                f"new:{self._instance_type or 'unknown'}@{self._instance_ip}"
            )
        else:
            instance_part = "unknown"

        gpu = self._instance_type or "unknown"
        summary = f"[summary] run={run_id} state={state} instance={instance_part} gpu={gpu}"
        if self._activity_start_ts:
            summary += f" activity_started={self._activity_start_ts}"
        if self._activity_end_ts:
            summary += f" activity_finished={self._activity_end_ts}"
        if self._results_end_ts:
            summary += f" results_ready={self._results_end_ts}"
        self.emit(summary)

        if agent_cost_usd is not None:
            self.emit(f"[summary] agent_cost=${agent_cost_usd:.4f}")

    # ------------------------------------------------------------------ #
    # Tool call formatting                                                  #
    # ------------------------------------------------------------------ #

    def _format_tool_call(self, tool_name: str, tool_input: dict) -> list[str]:
        """Return zero or more progress lines for a tool call."""
        # Strip MCP server prefix
        short = tool_name
        for prefix in ("mcp__lambda-repl__", "mcp__local-repl__", "mcp__docker-repl__"):
            if tool_name.startswith(prefix):
                short = tool_name[len(prefix):]
                break

        match short:
            # ── Standard file tools ──────────────────────────────────────────
            case "Read":
                return self._fmt_read(tool_input.get("file_path", ""))
            case "Write":
                p = tool_input.get("file_path", "")
                return [f"[setup] Write {Path(p).name}" if p else "[setup] Write file"]
            case "Edit":
                p = tool_input.get("file_path", "")
                return [f"[setup] Edit {Path(p).name}" if p else "[setup] Edit file"]
            case "Glob":
                return self._fmt_glob(
                    tool_input.get("pattern", ""),
                    tool_input.get("path", ""),
                )
            case "Grep":
                pat = tool_input.get("pattern", "")
                path = tool_input.get("path", "")
                return [f"[inspect] grep '{pat}'" + (f" in {path}" if path else "")]
            case "Bash":
                return self._fmt_bash(tool_input.get("command", ""))

            # ── Agent / schema tools ─────────────────────────────────────────
            case "Agent":
                sub = tool_input.get("subagent_type") or tool_input.get("description", "")
                return [f"[agent] Delegating to: {sub}" if sub else "[agent] Launching subagent"]
            case "ToolSearch":
                q = tool_input.get("query", "")
                return [f"[agent] Loading tool schema: {q}" if q else "[agent] Loading tool schema"]

            # ── Internal Claude Code task / plan tools (suppress or summarise)
            case "TodoWrite" | "TodoRead":
                return ["[agent] Updating task checklist"]
            case "TaskCreate" | "TaskUpdate" | "TaskList" | "TaskGet" | "TaskOutput" | "TaskStop":
                return ["[agent] Managing tasks"]
            case "ExitPlanMode" | "EnterPlanMode":
                return []  # suppress

            # ── Lambda MCP tools ─────────────────────────────────────────────
            case "lambda_preflight":
                return ["[lambda] Running preflight checks"]
            case "lambda_find_local_ssh_keys":
                return ["[lambda] Scanning local SSH keys"]
            case "lambda_list_ssh_keys":
                return ["[lambda] Listing remote SSH keys"]
            case "lambda_list_instance_types":
                return ["[lambda] Listing instance types"]
            case "lambda_provision_instance":
                itype = tool_input.get("instance_type", "")
                return [f"[lambda] Provisioning {itype}" if itype else "[lambda] Provisioning instance"]
            case "lambda_get_first_cost_estimate" | "lambda_get_cost_estimate":
                return ["[lambda] Estimating cost"]
            case "lambda_terminate_safe":
                iid = tool_input.get("instance_id", "")
                return [f"[lambda] Terminating instance {iid}" if iid else "[lambda] Terminating instance"]
            case "lambda_start_session":
                host = tool_input.get("host", "")
                if host:
                    self._instance_ip = host
                return [f"[lambda] Connecting to instance @ {host}" if host else "[lambda] Connecting to instance"]
            case "lambda_disconnect":
                return ["[lambda] Disconnecting session"]
            case "lambda_ensure_workspace":
                return ["[tmux] Ensuring local workspace"]
            case "lambda_ensure_remote_tmux":
                alias = tool_input.get("ssh_alias", "")
                return [f"[tmux] Ensuring remote session on {alias}" if alias else "[tmux] Ensuring remote session"]
            case "lambda_push_files":
                return ["[rsync] Pushing inputs to remote"]
            case "lambda_pull_files":
                return ["[rsync] Pulling results from remote"]
            case "lambda_upload_run":
                return ["[rsync] Uploading run inputs"]
            case "lambda_start_inference":
                run_id = tool_input.get("run_id", "")
                return [f"[{self._activity_tag}] Starting remote job" + (f" ({run_id})" if run_id else "")]
            case "lambda_sync_run_status":
                return [f"[{self._activity_tag}] Syncing run status"]
            case "lambda_read_run_status":
                return [f"[{self._activity_tag}] Reading run status"]
            case "lambda_get_output":
                return [f"[{self._activity_tag}] Polling remote output"]
            case "lambda_run_command":
                cmd = tool_input.get("command", "")
                return self._fmt_remote_cmd(self._instance_ip or "remote", cmd)
            case s if s.startswith("lambda_"):
                label = s[len("lambda_"):].replace("_", " ")
                return [f"[lambda] {label}"]

            # ── Local REPL ───────────────────────────────────────────────────
            case "local_repl_execute":
                cmd = tool_input.get("command", "")
                return [f"[shell] {cmd}" if cmd else "[shell] local command"]
            case _:
                return [f"[tool] {tool_name}"]

    def _fmt_read(self, path: str) -> list[str]:
        if not path:
            return ["[inspect] Read file"]
        if "skills/" in path:
            after = path.split("skills/")[-1]
            parts = [p for p in after.split("/") if p]
            skill_path = "/".join(parts[:2]) if len(parts) >= 2 else parts[0] if parts else after
            return [f"Read skill/{skill_path}"]
        return [f"[inspect] Read {Path(path).name}"]

    def _fmt_glob(self, pattern: str, base: str) -> list[str]:
        full = pattern or base
        if "skills/" in full:
            after = full.split("skills/")[-1].lstrip("/")
            fam = after.split("/")[0]
            return [f"[setup] Search skills/{fam}"]
        parts = [p for p in full.replace("\\", "/").split("/") if p]
        short = (".../" + "/".join(parts[-3:])) if len(parts) > 3 else full
        return [f"[inspect] Glob {short}"]

    @staticmethod
    def _describe_python_purpose(cmd: str) -> str:
        """Return a human-readable label for a python3 -c inline script."""
        c = cmd.lower()
        if "virtual_memory" in c or "psutil" in c:
            return "Checking available system memory"
        if "load_dataset" in c and any(k in c for k in ("counter", "label", "class", "unique")):
            return "python: computing label distribution"
        if "load_dataset" in c and any(k in c for k in ("len(", "percentile", "length")):
            return "python: computing sequence length statistics"
        if "load_dataset" in c and "plot" in c:
            return "python: generating dataset plots"
        if "load_dataset" in c or "datasets" in c:
            return "python: loading dataset"
        if "matplotlib" in c or "plt." in c or "savefig" in c:
            return "python: generating plots"
        if any(k in c for k in ("percentile", "histogram", "distribution")):
            return "python: computing statistics"
        if "json.dumps" in c or "json.loads" in c:
            return "python: processing JSON"
        if "hf_hub" in c or "huggingface" in c:
            return "python: querying HuggingFace Hub"
        return "python: running script"

    def _fmt_bash(self, cmd: str) -> list[str]:
        """Parse a Bash command string into one or more labelled progress lines."""
        if not cmd:
            return []

        raw = cmd.strip()

        # ── Pure comment ─────────────────────────────────────────────────────
        if raw.startswith("#"):
            first = raw.split("\n")[0].lstrip("#").strip()
            return ([f"[setup] # {first}"] if first else [])

        # ── Strip leading env boilerplate: `cd /x && export K=v && ...` ─────
        core = re.sub(
            r'^(?:cd\s+\S+\s*&&\s*)*'
            r'(?:export\s+\w+=(?:"[^"]*"|\'[^\']*\'|\S*)\s*&&\s*)*',
            "", raw,
        ).strip()

        # ── rsync ────────────────────────────────────────────────────────────
        if re.search(r"\brsync\b", raw):
            return self._fmt_rsync(raw)

        # ── SSH config update (heredoc >> ~/.ssh/config) ─────────────────────
        if ".ssh/config" in raw:
            return ["[lambda] Updating SSH config (~/.ssh/config)"]

        # ── Heredoc piped to SSH to write a remote file ──────────────────────
        if "<<" in raw and "ssh" in raw and ("> " in raw or "tee " in raw):
            m_file = re.search(r'(?:cat\s+>|tee)\s+["\']?([^\s"\'|]+)', raw)
            m_host = re.search(r"ubuntu@([\d.]+)", raw)
            rfile = Path(m_file.group(1)).name if m_file else "file"
            host = m_host.group(1) if m_host else "remote"
            return [f"[lambda] {host}: Writing {rfile}"]

        # ── ssh command ───────────────────────────────────────────────────────
        if re.match(r"^ssh\b", core):
            return self._fmt_ssh(raw)

        # ── sleep N && ... (polling heartbeat) ───────────────────────────────
        if re.match(r"^sleep\s+\d+", raw) and "&&" in raw:
            m = re.search(r"sleep\s+(\d+)", raw)
            n = m.group(1) if m else "?"
            rest = raw.split("&&", 1)[1].strip()
            purpose = _describe_poll_purpose(rest)
            return [
                f"[{self._activity_tag}] {purpose} in {n}s",
                f"[ssh.cmd] {rest}",
            ]

        # ── local mkdir ──────────────────────────────────────────────────────
        if re.match(r"^mkdir\b", core):
            m = re.search(r"mkdir\s+(?:-p\s+)?(.+)", core)
            dirs = m.group(1).strip() if m else core
            return [f"[setup] mkdir {dirs}"]

        # ── local cp / mv ─────────────────────────────────────────────────────
        if re.match(r"^(?:cp|mv)\b", core):
            parts = core.split()
            op = "Copy" if parts[0] == "cp" else "Move"
            src = Path(parts[1]).name if len(parts) > 1 else "?"
            dst = Path(parts[-1]).name if len(parts) > 2 else "?"
            return [f"[setup] {op} {src} → {dst}"]

        # ── pip install (local) ───────────────────────────────────────────────
        if "pip install" in raw:
            m = re.search(r"pip install\s+(?:-q\s+)?(.+?)(?:\s*&&|\s*2>|\s*$)", raw, re.DOTALL)
            pkgs = m.group(1).strip() if m else "packages"
            return [f"[setup] pip install {pkgs}"]

        # ── system / hardware resource queries ───────────────────────────────
        _resource_patterns = (
            r"vm_stat", r"free\s+-[mbg]", r"sysctl\s+hw\.mem",
            r"psutil\.virtual_memory", r"psutil\.disk_usage",
            r"/proc/meminfo", r"cat\s+/proc/meminfo",
        )
        if any(re.search(p, raw) for p in _resource_patterns):
            return ["[hardware] Checking system memory"]

        # ── disk / storage queries ────────────────────────────────────────────
        if re.match(r"^df\b", core):
            return ["[hardware] Checking disk space"]

        # ── local python ──────────────────────────────────────────────────────
        if re.match(r"python\d?(\s+-[a-zA-Z]+)*\s", core):
            # Inline -c script: derive a purpose label from key terms in the script
            if "-c" in core:
                purpose = self._describe_python_purpose(core)
                return [f"[setup] {purpose}"]
            # Script file
            m = re.search(r"python\d?(?:\s+-[a-zA-Z]+)*\s+(\S+)", core)
            script = Path(m.group(1)).name if m else ""
            return [f"[setup] python {script}"]

        # ── inspection commands ───────────────────────────────────────────────
        if re.match(r"^ls\b", core):
            return [f"[inspect] ls {core[3:].strip()}"]
        if re.match(r"^find\b", core):
            return [f"[inspect] find {core[5:].strip()}"]
        if re.match(r"^cat\b", core) and "<<" not in raw:
            return [f"[inspect] cat {core[4:].strip()}"]
        if re.match(r"^grep\b", core):
            return [f"[inspect] grep {core[5:].strip()}"]

        # ── fallback: show short summary, never the full raw command ──────────
        short = raw.split("\n")[0][:80]
        return [f"[shell] {short}{'…' if len(raw) > 80 else ''}"]

    def _fmt_rsync(self, raw: str) -> list[str]:
        """Format a rsync command with direction, source, and destination."""
        # Filter flag tokens; also strip shell redirects
        tokens = [
            t for t in raw.split()
            if not t.startswith("-")
            and t != "rsync"
            and not re.match(r"^\d*>>?", t)   # strip 2>&1, >, >> etc.
            and t not in ("2>&1", "&1")
        ]
        # Remove -e "..." arg value (value may be a single quoted string)
        clean_tokens: list[str] = []
        skip_next = False
        for t in tokens:
            if skip_next:
                skip_next = False
                continue
            if t == "-e":
                skip_next = True
                continue
            clean_tokens.append(t)

        src = clean_tokens[-2] if len(clean_tokens) >= 2 else ""
        dst = clean_tokens[-1] if len(clean_tokens) >= 1 else ""

        src_remote = ":" in src and not src.startswith("/") and not src.startswith("~")
        dst_remote = ":" in dst and not dst.startswith("/") and not dst.startswith("~")

        if not src_remote and dst_remote:
            # Push: local → remote
            host = dst.split(":")[0]
            rpath = dst.split(":")[1]
            if not self._activity_started:
                self.phase_start("rsync-push")
            return [
                f"[rsync] Pushing {src} → {host}:{rpath}",
                f"[rsync.cmd] {raw}",
            ]
        elif src_remote and not dst_remote:
            # Pull: remote → local
            host = src.split(":")[0]
            rpath = src.split(":")[1]
            return [
                f"[rsync] Pulling {host}:{rpath} → {dst}",
                f"[rsync.cmd] {raw}",
            ]
        # Fallback
        return [f"[rsync] {raw}"]

    def _fmt_ssh(self, raw: str) -> list[str]:
        """Parse an SSH command using _parse_ssh_target_and_remote."""
        target, remote = _parse_ssh_target_and_remote(raw)

        # Update tracked SSH target
        if target:
            self._ssh_target = target
            m_ip = re.search(r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", target)
            if m_ip:
                self._instance_ip = m_ip.group(1)

        host_label = target or self._instance_ip or "remote"

        # Pure connectivity probe
        if not remote or re.match(r"^(true|whoami)$", remote.strip()) or re.match(r"^echo\s+\S+$", remote.strip()):
            return [f"[ssh] Probing SSH availability: {host_label}"]

        # status.json check
        if "status.json" in remote:
            fname = remote.split()[-1].split("2>")[0].strip()
            fname = Path(fname).name
            # Set billing start for new instances on first real SSH work
            self._maybe_set_billing_start()
            return [
                f"[{self._activity_tag}] Checking run status: {fname} on {host_label}",
                f"[ssh.cmd] {raw}",
            ]

        # ls outputs/ check
        if re.match(r"ls\b", remote.strip()):
            path = remote.strip()[3:].strip().split()[0] if len(remote.strip()) > 3 else ""
            return [
                f"[results] Checking output files on {host_label}: {path}",
                f"[ssh.cmd] {raw}",
            ]

        # nvidia-smi
        if "nvidia-smi" in remote:
            return [f"[hardware] Querying GPU info on {host_label}"]

        # Set billing start for new instances on first real SSH work
        self._maybe_set_billing_start()

        # Delegate remaining cases to _fmt_remote_cmd
        lines = self._fmt_remote_cmd(host_label, remote)
        return lines

    def _maybe_set_billing_start(self) -> None:
        """
        Set billing start timestamp for new-instance runs on the first non-probe SSH call.
        For reused instances, billing start is set when the activity begins (in _fmt_remote_cmd).
        """
        if self._instance_reused is False and self._billing_start_t is None:
            self._billing_start_t = time.monotonic()
            self._billing_start_ts = _ts()

    def _fmt_remote_cmd(self, host: str, remote_cmd: str) -> list[str]:
        """Format a command executed on a remote host."""
        if not remote_cmd:
            return []

        # Strip leading cd/export boilerplate (handle both && and ;)
        core = re.sub(
            r'^(?:cd\s+\S+\s*(?:&&|;)\s*)*(?:export\s+\w+=\S*\s*(?:&&|;)\s*)*',
            "", remote_cmd.strip(),
        ).strip()

        # tmux
        if core.startswith("tmux "):
            if "new-session" in core:
                m = re.search(r"-s\s+(\S+)", core)
                sname = f" {m.group(1)}" if m else ""
                return [f"[tmux] {host}: Creating session{sname}"]
            if "has-session" in core:
                return [f"[tmux] {host}: Checking session existence"]
            if "send-keys" in core:
                return [f"[tmux] {host}: Sending keys to session"]
            return [f"[tmux] {host}: {core}"]

        # mkdir on remote
        if re.match(r"^mkdir\b", core):
            m = re.search(r"mkdir\s+(?:-p\s+)?(.+?)(?:\s*&&|\s*$)", core)
            dirs = m.group(1).strip() if m else core
            return [f"[lambda] {host}: mkdir {dirs}"]

        # nvidia-smi / GPU info
        if "nvidia-smi" in core:
            return [f"[hardware] {host}: Querying GPU info (nvidia-smi)"]

        # pip install on remote
        if "pip install" in core:
            m = re.search(r"pip install\s+(?:-[^\s]+\s+)*(.+?)(?:\s*&&|\s*2>|\s*$)", core, re.DOTALL)
            pkgs = m.group(1).strip() if m else "packages"
            return [
                f"[setup] {host}: pip install {pkgs}",
                f"[setup.cmd] {core}",
            ]

        # python / inference
        if re.search(r"python\d?\s", core):
            m = re.search(r"python\d?\s+(\S+)", core)
            script = m.group(1) if m else ""
            script_name = Path(script).name if script else script

            activity_keywords = _ACTIVITY_KEYWORDS.get(
                self._activity_tag, _INFERENCE_KEYWORDS
            )
            is_activity = any(kw in core.lower() for kw in activity_keywords)
            tag = self._activity_tag
            if is_activity and not self._activity_started:
                self._activity_started = True
                self._activity_t0 = time.monotonic()
                self._activity_start_ts = _ts()
                # Lazy reuse detection: if we reach the activity without a launch step
                if self._potential_reuse_ip and self._instance_reused is None:
                    self._instance_reused = True
                    self._ssh_target = host
                # Billing start for reused instances begins at activity time
                if self._instance_reused is True and self._billing_start_t is None:
                    self._billing_start_t = self._activity_t0
                    self._billing_start_ts = self._activity_start_ts
                return [
                    f"[{tag}] started_at={self._activity_start_ts} on {host}",
                    f"[{tag}] Running: python {script_name}",
                    f"[{tag}.cmd] {core}",
                ]
            if is_activity:
                return [
                    f"[{tag}] {host}: python {script_name}",
                    f"[{tag}.cmd] {core}",
                ]
            return [
                f"[lambda] {host}: python {script_name}",
                f"[lambda.cmd] {core}",
            ]

        # cat file (status/log reading)
        if re.match(r"^cat\b", core):
            target = core[4:].strip().split()[0] if len(core) > 4 else ""
            name = Path(target).name if target else "file"
            if any(kw in name for kw in ("status", "metrics", ".json", ".log")):
                return [f"[{self._activity_tag}] {host}: Reading {name}"]
            return [f"[inspect] {host}: cat {name}"]

        # ls on remote
        if re.match(r"^ls\b", core):
            return [f"[results] {host}: ls {core[3:].strip()}"]

        # grep on remote
        if re.match(r"^grep\b", core):
            return [f"[inspect] {host}: grep {core[5:].strip()}"]

        # write via printf/echo/tee
        if re.search(r"\bprintf\b|\btee\b", core):
            return [f"[lambda] {host}: Writing file"]

        # Generic remote command — show full text
        return [f"[lambda] {host}: {core}"]

    # ------------------------------------------------------------------ #
    # Tool result parsing                                                   #
    # ------------------------------------------------------------------ #

    def _parse_tool_result(self, tool_name: str, result: str) -> list[str]:
        """
        Extract meaningful follow-up lines from a tool's stdout/result text.
        Returns [] when there is nothing worth surfacing.
        """
        lines: list[str] = []
        if not result or not result.strip():
            return lines

        bash_cmd = self._last_bash_cmd if tool_name == "Bash" else ""

        # ── system resource check results ─────────────────────────────────────
        if tool_name == "Bash":
            _resource_patterns = (
                r"vm_stat", r"free\s+-[mbg]", r"sysctl\s+hw\.mem",
                r"psutil\.virtual_memory", r"psutil\.disk_usage",
            )
            if any(re.search(p, bash_cmd) for p in _resource_patterns):
                # Extract first float followed by GB/MB from the result
                m = re.search(r"([\d.]+)\s*(GB|MB|TB)", result, re.IGNORECASE)
                if m:
                    val, unit = m.group(1), m.group(2).upper()
                    lines.append(f"[hardware] Available RAM: {val} {unit}")
                return lines

        # ── Python stdout: emit progress lines the script printed ─────────────
        if tool_name == "Bash" and re.match(r"python\d?", bash_cmd.lstrip()):
            for ln in result.splitlines():
                ln = ln.strip()
                if not ln or ln.startswith("{") or ln.startswith("["):
                    continue
                # Emit lines that look like progress (contain stage/step markers
                # or numeric progress indicators, but skip tracebacks/warnings)
                if re.search(
                    r"(Stage|Step|Phase|Loading|Loaded|Computing|Computed|Plotting|"
                    r"Wrote|Written|Done|Error|rows|samples|labels|p50|p90|p99|\d+%)",
                    ln, re.IGNORECASE,
                ) and not re.search(r"(Traceback|Warning:|DeprecationWarning)", ln):
                    lines.append(f"[progress] {ln[:200]}")
            return lines

        # ── rsync stats ───────────────────────────────────────────────────────
        if tool_name == "Bash" and "rsync" in bash_cmd:
            m_bytes = re.search(r"Total transferred file size:\s*([\d,]+)", result)
            if m_bytes:
                nb = int(m_bytes.group(1).replace(",", ""))
                lines.append(f"[rsync] Transfer complete: {_format_size(nb)}")
                if "rsync-push" in self._phase_timers:
                    self.phase_end("rsync-push")

        # ── SSH echo / connection check ────────────────────────────────────────
        if tool_name == "Bash" and "echo ready" in bash_cmd and result.strip():
            ip_addr = self._instance_ip
            itype = f" ({self._instance_type})" if self._instance_type else ""
            lines.append(f"[lambda] SSH ready: {ip_addr}{itype}")

        # ── nvidia-smi: detect GPU type ───────────────────────────────────────
        if tool_name == "Bash" and "nvidia-smi" in bash_cmd:
            m_gpu = re.search(
                r"(A100|H100|A10G|A10|V100|L40|L4|T4|RTX\s*\d+\w*|B200|GH200)", result, re.IGNORECASE
            )
            if m_gpu:
                gpu = m_gpu.group(1).strip()
                self._instance_type = gpu
                lines.append(f"[hardware] GPU detected: {gpu}")

        # ── status.json result ────────────────────────────────────────────────
        if '"state"' in result and tool_name == "Bash":
            m_state = re.search(r'"state"\s*:\s*"(\w+)"', result)
            if m_state:
                state = m_state.group(1)
                if state == "completed" and self._activity_started:
                    self._activity_started = False
                    self._activity_end_ts = _ts()
                    dur = _format_duration(time.monotonic() - self._activity_t0) \
                          if self._activity_t0 else "?"
                    lines.append(
                        f"[{self._activity_tag}] finished_at={self._activity_end_ts} duration={dur} state=completed"
                    )
                elif state == "failed":
                    lines.append(f"[{self._activity_tag}] state=failed — check run_err.log")
                elif state == "running":
                    m_step  = re.search(r'"current_step"\s*:\s*(\d+)', result)
                    m_total = re.search(r'"total_steps"\s*:\s*(\d+)', result)
                    if m_step and m_total:
                        lines.append(
                            f"[{self._activity_tag}] Progress: {m_step.group(1)}/{m_total.group(1)} steps"
                        )

        # ── ls -lh output: format file sizes ─────────────────────────────────
        if tool_name == "Bash" and re.search(r"\bls\b.*-[lhla]", bash_cmd):
            for ln in result.splitlines():
                m = re.match(
                    r"^[-drwx]{10}\s+\d+\s+\w+\s+\w+\s+(\d+)\s+\w+\s+\d+\s+[\d:]+\s+(.+)$",
                    ln,
                )
                if m:
                    size = _format_size(int(m.group(1)))
                    fname = m.group(2).strip()
                    lines.append(f"[results] {fname}  {size}")

        return lines


# ---------------------------------------------------------------------------
# Context variable
# ---------------------------------------------------------------------------

_emitter: ContextVar[Optional[ProgressEmitter]] = ContextVar(
    "_capo_progress_emitter", default=None
)

# Runner tag — identifies the parallel pre-launch runner currently executing
# (e.g. "infra", "data", "model"). Each asyncio Task that sets this via
# set_runner_tag() has its own context copy, so concurrent runners are
# correctly isolated. When set, every emitted line is prefixed with [tag].
_runner_tag: ContextVar[str] = ContextVar("_capo_runner_tag", default="")


def set_emitter(
    emitter: Optional[ProgressEmitter],
) -> "Token[Optional[ProgressEmitter]]":
    """Install emitter in the current async/thread context. Returns a reset token."""
    return _emitter.set(emitter)


def set_runner_tag(tag: str) -> "Token[str]":
    """Tag all subsequent emissions in this async context with [tag]. Returns reset token."""
    return _runner_tag.set(tag)


def reset_runner_tag(token: "Token[str]") -> None:
    """Reset the runner tag to its previous value."""
    _runner_tag.reset(token)


def get_emitter() -> Optional[ProgressEmitter]:
    return _emitter.get()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def emit(message: str) -> None:
    """Emit a progress line. Delegates to context emitter or falls back to print()."""
    e = _emitter.get()
    if e is not None:
        e.emit(message)
    else:
        print(f"{_ts()} {message}", flush=True)


def error(message: str) -> None:
    """Emit an error line. Delegates to context emitter or falls back to stderr."""
    e = _emitter.get()
    if e is not None:
        e.error(message)
    else:
        print(f"{_ts()} {message}", file=sys.stderr, flush=True)


def emit_tool_call(tool_name: str, tool_input: dict) -> None:
    """Emit progress for a tool call. Delegates to context emitter."""
    e = _emitter.get()
    if e is not None:
        e.emit_tool_call(tool_name, tool_input)
    else:
        for line in _stateless_format(tool_name, tool_input):
            print(f"{_ts()} {line}", flush=True)


def emit_tool_result(tool_name: str, result_text: str) -> None:
    """Emit follow-up lines from a tool result. Delegates to context emitter."""
    e = _emitter.get()
    if e is not None:
        e.emit_tool_result(tool_name, result_text)


def _stateless_format(tool_name: str, tool_input: dict) -> list[str]:
    """Minimal formatter used when no ProgressEmitter is active."""
    short = tool_name
    for prefix in ("mcp__lambda-repl__", "mcp__local-repl__", "mcp__docker-repl__"):
        if tool_name.startswith(prefix):
            short = tool_name[len(prefix):]
            break
    if short == "Bash":
        cmd = tool_input.get("command", "").strip()
        return [f"[shell] {cmd[:200]}"] if cmd else []
    if short == "Read":
        p = tool_input.get("file_path", "")
        if "skills/" in p:
            after = p.split("skills/")[-1]
            return [f"Read skill/{'/'.join(after.split('/')[:2])}"]
        return [f"[inspect] Read {Path(p).name}"]
    if short == "Write":
        return [f"[setup] Write {Path(tool_input.get('file_path', '')).name}"]
    if short == "Glob":
        return [f"[inspect] Glob {tool_input.get('pattern', '')}"]
    if short == "Grep":
        return [f"[inspect] grep '{tool_input.get('pattern', '')}'"]
    if short.startswith("lambda_"):
        return [f"[lambda] {short[7:].replace('_', ' ')}"]
    return [f"[tool] {tool_name}"]


# ---------------------------------------------------------------------------
# Convenience: human-readable file sizes (exported for use in result summaries)
# ---------------------------------------------------------------------------

format_size = _format_size
format_duration = _format_duration

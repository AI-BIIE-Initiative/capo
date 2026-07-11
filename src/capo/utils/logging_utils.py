import gzip
import hashlib
import json
import os
import re
import socket
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
LOG_ROOT = REPO_ROOT / "logs"
MAX_JSONL_BYTES = 50 * 1024 * 1024
MAX_FIELD_CHARS = 16384
PREVIEW_LINES = 3


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_iso(dt: datetime) -> str:
    return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _epoch_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _git_commit(cwd: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=cwd,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        return None
    return None


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _truncate_text(text: str, max_chars: int = MAX_FIELD_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...<truncated>"


def _preview_text(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    head = lines[:PREVIEW_LINES]
    tail = lines[-PREVIEW_LINES:] if len(lines) > PREVIEW_LINES else []
    return {
        "head": head,
        "tail": tail,
        "line_count": len(lines),
    }


def _redact_string(value: str) -> str:
    patterns = [
        r"sk-[A-Za-z0-9_-]{20,}",
        r"pk-[A-Za-z0-9_-]{20,}",
        r"[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}",
    ]
    for pattern in patterns:
        value = re.sub(pattern, "[REDACTED]", value)
    return value


def _redact(obj: Any) -> Any:
    if isinstance(obj, dict):
        redacted: dict[str, Any] = {}
        for key, val in obj.items():
            key_upper = str(key).upper()
            if any(k in key_upper for k in ["KEY", "TOKEN", "SECRET", "PASSWORD"]):
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = _redact(val)
        return redacted
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    if isinstance(obj, str):
        return _redact_string(obj)
    return obj


class JSONLWriter:
    def __init__(self, path_prefix: str, use_current_suffix: bool = True):
        self._path_prefix = path_prefix
        self._current_path = (
            f"{path_prefix}.current.jsonl"
            if use_current_suffix
            else f"{path_prefix}.jsonl"
        )
        self._lock = threading.Lock()
        self._index = 0

        _ensure_dir(os.path.dirname(self._current_path))

    def _rotate_if_needed(self) -> None:
        if not os.path.exists(self._current_path):
            return
        if os.path.getsize(self._current_path) < MAX_JSONL_BYTES:
            return

        gz_path = f"{self._path_prefix}.{self._index}.jsonl.gz"
        self._index += 1
        with open(self._current_path, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
            f_out.writelines(f_in)
        os.remove(self._current_path)

    def append(self, obj: dict[str, Any]) -> None:
        line = json.dumps(obj, ensure_ascii=False)
        with self._lock:
            self._rotate_if_needed()
            with open(self._current_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


@dataclass
class RunContext:
    server_name: str
    run_id: str
    run_dir: str
    tool_writer: JSONLWriter
    trace_writer: JSONLWriter
    payload_writer: JSONLWriter
    payload_dir: str
    sessions_dir: str
    run_meta: dict[str, Any]

    def write_payload(self, call_id: str, kind: str, content: str) -> str:
        record = {
            "schema_version": "1.0",
            "call_id": call_id,
            "kind": kind,
            "ts": _utc_iso(_utc_now()),
            "content": content,
        }
        self.payload_writer.append(record)
        return "payloads/payloads.jsonl"

    def write_json_payload(self, call_id: str, kind: str, payload: Any) -> str:
        filename = f"{call_id}_{kind}.json.gz"
        path = os.path.join(self.payload_dir, filename)
        _ensure_dir(self.payload_dir)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return f"payloads/{filename}"

    def log_tool_call(self, event: dict[str, Any]) -> None:
        self.tool_writer.append(event)

    def log_trace(self, event: dict[str, Any]) -> None:
        self.trace_writer.append(event)

    def update_session(self, session_id: str, update: dict[str, Any]) -> None:
        _ensure_dir(self.sessions_dir)
        path = os.path.join(self.sessions_dir, f"{session_id}.json")
        data: dict[str, Any] = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        data.update(update)
        data.setdefault("session_id", session_id)
        data.setdefault("run_id", self.run_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


def init_run(server_name: str) -> RunContext:
    run_id = uuid.uuid4().hex
    dt = _utc_now()
    run_name = f"run_{dt.strftime('%Y%m%d_%H%M%S')}_{run_id}"
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    run_dir = os.path.join(LOG_ROOT, "agent_tool_calls", server_name, run_name)
    payload_dir = os.path.join(run_dir, "payloads")
    trace_dir = os.path.join(run_dir, "traces")
    sessions_dir = os.path.join(LOG_ROOT, "sessions")

    _ensure_dir(payload_dir)
    _ensure_dir(trace_dir)

    run_meta = {
        "run_id": run_id,
        "server": server_name,
        "ts_start": _utc_iso(dt),
        "ts_epoch_ms": _epoch_ms(dt),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "python": os.sys.version.split()[0],
        "cwd": os.getcwd(),
        "git_commit": _git_commit(os.getcwd()),
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    tool_writer = JSONLWriter(os.path.join(run_dir, "tool_calls"))
    trace_writer = JSONLWriter(os.path.join(trace_dir, "trace_events"))
    payload_writer = JSONLWriter(
        os.path.join(payload_dir, "payloads"), use_current_suffix=False
    )

    return RunContext(
        server_name=server_name,
        run_id=run_id,
        run_dir=run_dir,
        tool_writer=tool_writer,
        trace_writer=trace_writer,
        payload_writer=payload_writer,
        payload_dir=payload_dir,
        sessions_dir=sessions_dir,
        run_meta=run_meta,
    )


def sanitize_input(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    safe = _redact(args)
    if tool_name in {
        "write_local_file",
        "docker_repl_write_file",
        "container_write_file",
    }:
        content = args.get("content", "")
        if isinstance(content, str):
            safe["content"] = {
                "size": len(content.encode("utf-8", errors="replace")),
                "sha256": _hash_text(content),
                "preview": _preview_text(content),
            }
    return safe


def sanitize_output(tool_name: str, output: Any) -> Any:
    if tool_name in {"read_local_file", "docker_repl_read_file", "container_read_file"}:
        if isinstance(output, dict) and "content" in output:
            content = output.get("content", "")
            if isinstance(content, str):
                output = dict(output)
                output["content"] = {
                    "size": len(content.encode("utf-8", errors="replace")),
                    "sha256": _hash_text(content),
                    "preview": _preview_text(content),
                }
    return _redact(output)


def log_tool_call(
    ctx: RunContext,
    tool_name: str,
    session_id: Optional[str],
    input_args: dict[str, Any],
    output: Any = None,
    error: Optional[str] = None,
    exception_text: Optional[str] = None,
    meta: Optional[dict[str, Any]] = None,
    started: Optional[datetime] = None,
) -> None:
    start_dt = started or _utc_now()
    end_dt = _utc_now()
    duration_ms = _epoch_ms(end_dt) - _epoch_ms(start_dt)
    call_id = uuid.uuid4().hex

    safe_input = sanitize_input(tool_name, input_args)

    stdout_ref = None
    stderr_ref = None
    result_ref = None
    exception_ref = None

    if isinstance(output, dict):
        if isinstance(output.get("stdout"), str):
            stdout_ref = ctx.write_payload(call_id, "stdout", output["stdout"])
            stdout_ref = f"{stdout_ref}#{call_id}:stdout"
            if len(output["stdout"]) > MAX_FIELD_CHARS:
                output = dict(output)
                output["stdout"] = _truncate_text(output["stdout"])
        if isinstance(output.get("stderr"), str):
            stderr_ref = ctx.write_payload(call_id, "stderr", output["stderr"])
            stderr_ref = f"{stderr_ref}#{call_id}:stderr"
            if len(output["stderr"]) > MAX_FIELD_CHARS:
                output = dict(output)
                output["stderr"] = _truncate_text(output["stderr"])

    safe_output = sanitize_output(tool_name, output) if output is not None else None

    # Avoid writing per-call result payloads unless needed

    if exception_text:
        exception_ref = ctx.write_payload(call_id, "exception", exception_text)

    event = {
        "schema_version": "1.0",
        "run_id": ctx.run_id,
        "server": ctx.server_name,
        "session_id": session_id,
        "call_id": call_id,
        "tool_name": tool_name,
        "ts_start": _utc_iso(start_dt),
        "ts_end": _utc_iso(end_dt),
        "duration_ms": duration_ms,
        "status": "error" if error or exception_text else "ok",
        "input": safe_input,
        "output": safe_output,
        "stdout_ref": stdout_ref,
        "stderr_ref": stderr_ref,
        "result_ref": result_ref,
        "exception_ref": exception_ref,
        "meta": meta or {},
    }

    ctx.log_tool_call(event)

    if session_id:
        existing_calls = 0
        path = os.path.join(ctx.sessions_dir, f"{session_id}.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing_calls = json.load(f).get("tool_call_count", 0)
            except Exception:
                existing_calls = 0

        ctx.update_session(
            session_id,
            {
                "last_activity": time.time(),
                "tool_call_count": existing_calls + 1,
            },
        )


def log_trace_event(
    ctx: RunContext,
    name: str,
    session_id: Optional[str],
    payload: dict[str, Any],
) -> None:
    event = {
        "schema_version": "1.0",
        "run_id": ctx.run_id,
        "server": ctx.server_name,
        "session_id": session_id,
        "event": name,
        "ts": _utc_iso(_utc_now()),
        "payload": _redact(payload),
    }
    ctx.log_trace(event)


def logged_tool(ctx: RunContext, tool_name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            started = _utc_now()
            session_id = None
            input_args = kwargs.copy()
            if "session_id" in input_args:
                session_id = input_args.get("session_id")

            output: Any = None
            original_output: Any = None
            error: Optional[str] = None
            exception_text: Optional[str] = None
            try:
                original_output = func(*args, **kwargs)
                output = original_output
                if isinstance(original_output, str):
                    try:
                        parsed = json.loads(original_output)
                        output = parsed
                    except Exception:
                        pass
                if tool_name.startswith("start_") and isinstance(output, dict):
                    session_id = session_id or output.get("session_id")
            except Exception as e:
                error = str(e)
                exception_text = repr(e)
                raise
            finally:
                meta = {
                    "pid": os.getpid(),
                    "cwd": os.getcwd(),
                    "host": socket.gethostname(),
                    "python": os.sys.version.split()[0],
                }
                if "image" in input_args:
                    meta["image"] = input_args.get("image")
                if "container_id" in input_args:
                    meta["container_id"] = input_args.get("container_id")

                log_tool_call(
                    ctx=ctx,
                    tool_name=tool_name,
                    session_id=session_id,
                    input_args=input_args,
                    output=output,
                    error=error,
                    exception_text=exception_text,
                    meta=meta,
                    started=started,
                )

            return original_output

        wrapper.__name__ = tool_name
        return wrapper

    return decorator

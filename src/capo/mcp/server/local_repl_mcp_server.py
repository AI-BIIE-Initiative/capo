from __future__ import annotations

import json
import os
import secrets
import time
from typing import Any

from mcp.server.fastmcp import FastMCP

from capo.mcp.environments.local_repl_env import LocalREPL
from capo.utils.logging_utils import init_run, logged_tool, log_trace_event

mcp = FastMCP("local-repl")

RUN_CTX = init_run("local_repl_mcp_server")

BASE_DIR = os.getcwd()


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


_SESSIONS: dict[str, LocalREPL] = {}


def _safe_path(relative_path: str) -> str:
    if os.path.isabs(relative_path):
        abs_path = os.path.abspath(relative_path)
    else:
        rel = relative_path.lstrip("/\\")
        abs_path = os.path.abspath(os.path.join(BASE_DIR, rel))
    base = os.path.abspath(BASE_DIR)
    if abs_path == base or abs_path.startswith(base + os.sep):
        return abs_path
    raise ValueError("Path escapes workspace root")


@mcp.tool()
@logged_tool(RUN_CTX, "start_local_repl_session")
def start_local_repl_session(
    context_payload: dict | list | str | None = None,
    setup_code: str | None = None,
    depth: int = 1,
) -> str:
    """
    Start a persistent local Python REPL session.

    Output JSON contains: ok, session_id, created_at, note.
    """
    session_id = f"local_repl_{secrets.token_hex(8)}"
    repl = LocalREPL(
        context_payload=context_payload,
        setup_code=setup_code,
        persistent=True,
        work_dir=BASE_DIR,
        depth=depth,
    )
    repl.set_logger(RUN_CTX, session_id=session_id)
    _SESSIONS[session_id] = repl

    RUN_CTX.update_session(
        session_id,
        {
            "status": "started",
            "started_at": time.time(),
            "last_activity": time.time(),
        },
    )
    log_trace_event(
        RUN_CTX,
        "session_start",
        session_id,
        {"env": "local", "session_id": session_id},
    )

    return _json(
        {
            "ok": True,
            "session_id": session_id,
            "created_at": time.time(),
            "note": "Call local_repl_execute(session_id, code) repeatedly; call stop_local_repl_session(session_id) to end.",
        }
    )


@mcp.tool()
@logged_tool(RUN_CTX, "write_local_file")
def write_local_file(relative_path: str, content: str, overwrite: bool = True) -> str:
    """
    Write a text file to the workspace root.

    Output JSON contains: ok, path, bytes_written.
    """
    try:
        abs_path = _safe_path(relative_path)
    except Exception as e:
        return _json({"ok": False, "error": str(e)})

    if not overwrite and os.path.exists(abs_path):
        return _json({"ok": False, "error": "File already exists"})

    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(content)

    return _json(
        {"ok": True, "path": abs_path, "bytes_written": len(content.encode("utf-8"))}
    )


@mcp.tool()
@logged_tool(RUN_CTX, "read_local_file")
def read_local_file(relative_path: str) -> str:
    """
    Read a text file from the workspace root.

    Output JSON contains: ok, path, content.
    """
    try:
        abs_path = _safe_path(relative_path)
    except Exception as e:
        return _json({"ok": False, "error": str(e)})

    if not os.path.exists(abs_path):
        return _json({"ok": False, "error": "File not found"})

    with open(abs_path, "r", encoding="utf-8") as f:
        content = f.read()

    return _json({"ok": True, "path": abs_path, "content": content})


@mcp.tool()
@logged_tool(RUN_CTX, "list_local_files")
def list_local_files(relative_dir: str = ".") -> str:
    """
    List files in a workspace subdirectory.

    Output JSON contains: ok, path, entries.
    """
    try:
        abs_path = _safe_path(relative_dir)
    except Exception as e:
        return _json({"ok": False, "error": str(e)})

    if not os.path.isdir(abs_path):
        return _json({"ok": False, "error": "Directory not found"})

    entries = sorted(os.listdir(abs_path))
    return _json({"ok": True, "path": abs_path, "entries": entries})


@mcp.tool()
@logged_tool(RUN_CTX, "local_repl_execute_file")
def local_repl_execute_file(session_id: str, relative_path: str) -> str:
    """
    Execute a local file by reading it into the REPL session.

    Output JSON contains: ok, stdout, stderr, execution_time.
    """
    repl = _SESSIONS.get(session_id)
    if not repl:
        return _json(
            {"ok": False, "stdout": "", "stderr": f"Unknown session_id: {session_id}"}
        )

    try:
        abs_path = _safe_path(relative_path)
    except Exception as e:
        return _json({"ok": False, "stdout": "", "stderr": str(e)})

    if not os.path.exists(abs_path):
        return _json({"ok": False, "stdout": "", "stderr": "File not found"})

    with open(abs_path, "r", encoding="utf-8") as f:
        code = f.read()

    result = repl.execute_code(code)
    return _json(
        {
            "ok": True,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "execution_time": result.execution_time,
        }
    )


@mcp.tool()
@logged_tool(RUN_CTX, "local_repl_execute")
def local_repl_execute(session_id: str, code: str) -> str:
    """
    Execute Python code inside an existing local REPL session.

    Output JSON contains: ok, stdout, stderr, execution_time.
    """
    repl = _SESSIONS.get(session_id)
    if not repl:
        return _json(
            {"ok": False, "stdout": "", "stderr": f"Unknown session_id: {session_id}"}
        )

    result = repl.execute_code(code)
    return _json(
        {
            "ok": True,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "execution_time": result.execution_time,
        }
    )


@mcp.tool()
@logged_tool(RUN_CTX, "stop_local_repl_session")
def stop_local_repl_session(session_id: str) -> str:
    """
    Stop a local REPL session and clean up its resources.
    """
    repl = _SESSIONS.pop(session_id, None)
    if not repl:
        return _json({"ok": False, "message": f"No such session: {session_id}"})

    try:
        repl.cleanup()
    except Exception as e:
        return _json({"ok": False, "message": f"Error stopping session: {e}"})

    RUN_CTX.update_session(
        session_id,
        {"status": "stopped", "ended_at": time.time(), "last_activity": time.time()},
    )
    log_trace_event(
        RUN_CTX,
        "session_end",
        session_id,
        {"env": "local", "session_id": session_id},
    )
    return _json({"ok": True, "message": f"Stopped session {session_id}"})


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

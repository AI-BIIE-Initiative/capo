from __future__ import annotations

import datetime
import fnmatch
import glob
import json
import os
import secrets
import shutil
import subprocess
import time
from typing import Any
import shlex

from mcp.server.fastmcp import FastMCP

from capo.mcp.environments.docker_repl_env import DockerREPL
from capo.utils.logging_utils import init_run, logged_tool, log_trace_event

DEFAULT_IMAGE = os.getenv("MCP_REPL_IMAGE", "autoimmunolab-repl:latest")


mcp = FastMCP("docker-repl")

RUN_CTX = init_run("docker_repl_mcp_server")

BASE_DIR = os.getcwd()
HOST_WORKSPACE = BASE_DIR


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _safe_path(base_dir: str, relative_path: str) -> str:
    if os.path.isabs(relative_path):
        abs_path = os.path.abspath(relative_path)
    else:
        rel = relative_path.lstrip("/\\")
        abs_path = os.path.abspath(os.path.join(base_dir, rel))
    base = os.path.abspath(base_dir)
    if abs_path == base or abs_path.startswith(base + os.sep):
        return abs_path
    raise ValueError("Path escapes workspace root")


def _resolve_container_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join("/workspace", path)


def _safe_host_path(relative_path: str) -> str:
    if os.path.isabs(relative_path):
        abs_path = os.path.abspath(relative_path)
    else:
        rel = relative_path.lstrip("/\\")
        abs_path = os.path.abspath(os.path.join(HOST_WORKSPACE, rel))
    base = os.path.abspath(HOST_WORKSPACE)
    if abs_path == base or abs_path.startswith(base + os.sep):
        return abs_path
    raise ValueError("Path escapes host workspace root")


def _exec_in_container(
    container_id_or_name: str, command: list[str], timeout_seconds: int = 30
) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["docker", "exec", container_id_or_name, *command],
            capture_output=True,
            text=True,
            timeout=int(timeout_seconds),
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "stdout": "",
            "stderr": "Command timed out",
            "exit_code": None,
        }
    except Exception as e:
        return {"ok": False, "stdout": "", "stderr": str(e), "exit_code": None}

    return {
        "ok": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.returncode,
    }


def _list_container_files(
    container_id_or_name: str,
) -> tuple[list[str], dict[str, float]]:
    """Return list of files (relative to /workspace) and mtime map if available."""
    files: list[str] = []
    mtimes: dict[str, float] = {}

    result = _exec_in_container(
        container_id_or_name,
        ["sh", "-c", "cd /workspace && find . -type f -printf '%P\t%T@\n'"],
    )
    if not result.get("ok"):
        result = _exec_in_container(
            container_id_or_name,
            ["sh", "-c", "cd /workspace && find . -type f -print"],
        )
        if not result.get("ok"):
            return files, mtimes
        for line in result.get("stdout", "").splitlines():
            line = line.strip()
            if line:
                files.append(line.lstrip("./"))
        return files, mtimes

    for line in result.get("stdout", "").splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        rel, mtime = parts
        rel = rel.lstrip("./")
        if not rel:
            continue
        files.append(rel)
        try:
            mtimes[rel] = float(mtime)
        except ValueError:
            continue

    return files, mtimes


_SESSIONS: dict[str, DockerREPL] = {}
_ACTIVE_CONTAINER: str | None = None
_SESSION_EXPORT_STATE: dict[str, float] = {}
_CONTAINER_EXPORT_STATE: dict[str, float] = {}


@mcp.tool()
@logged_tool(RUN_CTX, "set_active_container")
def set_active_container(container_id_or_name: str) -> str:
    """
    Set the active container used for container_* operations.
    """
    global _ACTIVE_CONTAINER
    _ACTIVE_CONTAINER = container_id_or_name
    return _json({"ok": True, "active_container": _ACTIVE_CONTAINER})


@mcp.tool()
@logged_tool(RUN_CTX, "set_active_container_by_latest_workspace")
def set_active_container_by_latest_workspace() -> str:
    """
    Set active container to the one with latest /workspace modification time.
    """
    payload = json.loads(list_running_containers(sort_by="modified"))
    if not payload.get("ok"):
        return _json(
            {"ok": False, "error": payload.get("error", "Failed to list containers")}
        )
    containers = payload.get("containers", [])
    if not containers:
        return _json({"ok": False, "error": "No running containers"})

    target = containers[0].get("id") or containers[0].get("name")
    if not target:
        return _json({"ok": False, "error": "No container id found"})

    return set_active_container(target)


@mcp.tool()
@logged_tool(RUN_CTX, "get_active_container")
def get_active_container() -> str:
    """
    Get the currently active container for container_* operations.
    """
    return _json({"ok": True, "active_container": _ACTIVE_CONTAINER})


@mcp.tool()
@logged_tool(RUN_CTX, "list_docker_images")
def list_docker_images() -> str:
    """
    List Docker images available on the host.

    Output JSON contains: ok, stdout, stderr, exit_code.
    """
    ready, msg = _docker_ready()
    if not ready:
        return _json({"ok": False, "error": msg})

    try:
        result = subprocess.run(
            ["docker", "images"],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except subprocess.TimeoutExpired:
        return _json(
            {
                "ok": False,
                "stdout": "",
                "stderr": "Command timed out",
                "exit_code": None,
            }
        )
    except Exception as e:
        return _json({"ok": False, "stdout": "", "stderr": str(e), "exit_code": None})

    return _json(
        {
            "ok": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
        }
    )


@mcp.tool()
@logged_tool(RUN_CTX, "list_running_containers")
def list_running_containers(sort_by: str = "created") -> str:
    """
    List running Docker containers on the host.

    Output JSON contains: ok, containers.
    """
    ready, msg = _docker_ready()
    if not ready:
        return _json({"ok": False, "error": msg})

    try:
        result = subprocess.run(
            [
                "docker",
                "ps",
                "--format",
                "{{.ID}}\t{{.Image}}\t{{.Names}}\t{{.CreatedAt}}",
            ],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except subprocess.TimeoutExpired:
        return _json(
            {
                "ok": False,
                "stdout": "",
                "stderr": "Command timed out",
                "exit_code": None,
            }
        )
    except Exception as e:
        return _json({"ok": False, "stdout": "", "stderr": str(e), "exit_code": None})

    if result.returncode != 0:
        return _json(
            {
                "ok": False,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
            }
        )

    containers: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 4:
            containers.append(
                {
                    "id": parts[0],
                    "image": parts[1],
                    "name": parts[2],
                    "created_at": parts[3],
                }
            )

    for container in containers:
        cid = container.get("id")
        if not cid:
            continue
        try:
            stat_result = subprocess.run(
                [
                    "docker",
                    "exec",
                    cid,
                    "sh",
                    "-c",
                    "stat -c %Y /workspace 2>/dev/null || stat -f %m /workspace",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if stat_result.returncode == 0:
                raw_mtime = stat_result.stdout.strip()
                if raw_mtime.isdigit():
                    mtime_int = int(raw_mtime)
                    container["workspace_mtime"] = mtime_int
                    container["workspace_mtime_iso"] = (
                        datetime.datetime.fromtimestamp(
                            mtime_int, tz=datetime.timezone.utc
                        )
                        .isoformat(timespec="seconds")
                        .replace("+00:00", "Z")
                    )
                else:
                    container["workspace_mtime"] = raw_mtime
        except Exception:
            continue

    if sort_by == "created":
        containers.sort(key=lambda x: x.get("created_at", ""))
    elif sort_by == "modified":
        containers.sort(key=lambda x: x.get("workspace_mtime", 0), reverse=True)

    return _json({"ok": True, "containers": containers})


def _docker_ready() -> tuple[bool, str]:
    from shutil import which

    if which("docker") is None:
        return False, "Docker CLI (`docker`) not found on PATH."

    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=6,
        )
    except subprocess.TimeoutExpired:
        return False, "Docker daemon did not respond (timeout)."
    except Exception as e:
        return False, f"Docker check failed: {e}"

    if result.returncode == 0:
        return True, "Docker is available."

    msg = (result.stderr or result.stdout).strip() or "Docker daemon not reachable."
    help_text = (
        "Docker is installed, but the daemon is not reachable.\n\n"
        "Typical fixes:\n"
        "- macOS/Windows: start Docker Desktop\n"
        "- Linux (systemd): `sudo systemctl start docker` then `sudo systemctl enable docker`\n"
        "- Ensure your user can access Docker (often: add user to `docker` group, then log out/in)\n\n"
        f"Raw error:\n{msg}"
    )
    return False, help_text


@mcp.tool()
@logged_tool(RUN_CTX, "start_docker_repl_session")
def start_docker_repl_session(
    image: str = DEFAULT_IMAGE,
    depth: int = 1,
) -> str:
    """
    Start a Docker-backed Python REPL session.

    Output JSON contains: ok, session_id, image, created_at, note.
    """
    ready, msg = _docker_ready()
    if not ready:
        return _json({"ok": False, "error": msg})

    session_id = f"docker_repl_{secrets.token_hex(8)}"
    try:
        repl = DockerREPL(
            image=image,
            depth=depth,
        )
    except Exception as e:
        return _json({"ok": False, "error": str(e)})

    repl.set_logger(RUN_CTX, session_id=session_id)
    _SESSIONS[session_id] = repl

    RUN_CTX.update_session(
        session_id,
        {
            "status": "started",
            "started_at": time.time(),
            "last_activity": time.time(),
            "image": image,
            "container_id": repl.container_id,
        },
    )
    log_trace_event(
        RUN_CTX,
        "session_start",
        session_id,
        {
            "env": "docker",
            "session_id": session_id,
            "image": image,
            "container_id": repl.container_id,
        },
    )

    return _json(
        {
            "ok": True,
            "session_id": session_id,
            "image": image,
            "created_at": time.time(),
            "container_id": repl.container_id,
            "note": "Call docker_repl_execute(session_id, code) repeatedly; call stop_docker_repl_session(session_id) to end.",
        }
    )


@mcp.tool()
@logged_tool(RUN_CTX, "docker_repl_execute")
def docker_repl_execute(session_id: str, code: str) -> str:
    """
    Execute Python code inside an existing Docker REPL session.

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
@logged_tool(RUN_CTX, "docker_repl_write_file")
def docker_repl_write_file(
    session_id: str, relative_path: str, content: str, overwrite: bool = True
) -> str:
    """
    Write a text file into the container workspace.

    Output JSON contains: ok, host_path, container_path.
    """
    repl = _SESSIONS.get(session_id)
    if not repl:
        return _json({"ok": False, "error": f"Unknown session_id: {session_id}"})

    try:
        abs_path = _safe_path(repl.temp_dir, relative_path)
    except Exception as e:
        return _json({"ok": False, "error": str(e)})

    if not overwrite and os.path.exists(abs_path):
        return _json({"ok": False, "error": "File already exists"})

    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(content)

    container_path = os.path.join("/workspace", relative_path.lstrip("/\\"))
    return _json({"ok": True, "host_path": abs_path, "container_path": container_path})


@mcp.tool()
@logged_tool(RUN_CTX, "docker_repl_execute_file")
def docker_repl_execute_file(session_id: str, relative_path: str) -> str:
    """
    Execute a file that exists in the container workspace.

    Output JSON contains: ok, stdout, stderr, execution_time.
    """
    repl = _SESSIONS.get(session_id)
    if not repl:
        return _json(
            {"ok": False, "stdout": "", "stderr": f"Unknown session_id: {session_id}"}
        )

    try:
        abs_path = _safe_path(repl.temp_dir, relative_path)
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
@logged_tool(RUN_CTX, "docker_repl_read_file")
def docker_repl_read_file(session_id: str, relative_path: str) -> str:
    """
    Read a text file from the container workspace.

    Output JSON contains: ok, host_path, container_path, content.
    """
    repl = _SESSIONS.get(session_id)
    if not repl:
        return _json({"ok": False, "error": f"Unknown session_id: {session_id}"})

    try:
        abs_path = _safe_path(repl.temp_dir, relative_path)
    except Exception as e:
        return _json({"ok": False, "error": str(e)})

    if not os.path.exists(abs_path):
        return _json({"ok": False, "error": "File not found"})

    with open(abs_path, "r", encoding="utf-8") as f:
        content = f.read()

    container_path = os.path.join("/workspace", relative_path.lstrip("/\\"))
    return _json(
        {
            "ok": True,
            "host_path": abs_path,
            "container_path": container_path,
            "content": content,
        }
    )


@mcp.tool()
@logged_tool(RUN_CTX, "docker_repl_run_command")
def docker_repl_run_command(
    session_id: str, command: list[str], timeout_seconds: int = 60
) -> str:
    """
    Run a command inside the Docker container (e.g., ["python3", "script.py", "--k", "8"]).

    Output JSON contains: ok, stdout, stderr, exit_code.
    """
    repl = _SESSIONS.get(session_id)
    if not repl:
        return _json({"ok": False, "error": f"Unknown session_id: {session_id}"})

    if not repl.container_id:
        return _json({"ok": False, "error": "Container not running"})

    if not command or not isinstance(command, list):
        return _json({"ok": False, "error": "Command must be a list of strings"})

    log_trace_event(
        RUN_CTX,
        "command_run_start",
        session_id,
        {
            "command": command,
            "container_id": repl.container_id,
            "timeout_seconds": timeout_seconds,
        },
    )

    try:
        result = subprocess.run(
            ["docker", "exec", repl.container_id, *command],
            capture_output=True,
            text=True,
            timeout=int(timeout_seconds),
        )
    except subprocess.TimeoutExpired:
        log_trace_event(
            RUN_CTX,
            "command_run_end",
            session_id,
            {
                "command": command,
                "container_id": repl.container_id,
                "error": "Command timed out",
            },
        )
        return _json(
            {
                "ok": False,
                "stdout": "",
                "stderr": "Command timed out",
                "exit_code": None,
            }
        )
    except Exception as e:
        log_trace_event(
            RUN_CTX,
            "command_run_end",
            session_id,
            {
                "command": command,
                "container_id": repl.container_id,
                "error": str(e),
            },
        )
        return _json({"ok": False, "stdout": "", "stderr": str(e), "exit_code": None})

    log_trace_event(
        RUN_CTX,
        "command_run_end",
        session_id,
        {
            "command": command,
            "container_id": repl.container_id,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
        },
    )

    return _json(
        {
            "ok": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
        }
    )


@mcp.tool()
@logged_tool(RUN_CTX, "container_run_command")
def container_run_command(
    container_id_or_name: str, command: list[str], timeout_seconds: int = 30
) -> str:
    """
    Run a command inside a specific running container.

    Output JSON contains: ok, stdout, stderr, exit_code.
    """
    ready, msg = _docker_ready()
    if not ready:
        return _json({"ok": False, "error": msg})

    if not command or not isinstance(command, list):
        return _json({"ok": False, "error": "Command must be a list of strings"})

    log_trace_event(
        RUN_CTX,
        "command_run_start",
        None,
        {
            "command": command,
            "container_id": container_id_or_name,
            "timeout_seconds": timeout_seconds,
        },
    )

    result = _exec_in_container(
        container_id_or_name, command, timeout_seconds=timeout_seconds
    )
    log_trace_event(
        RUN_CTX,
        "command_run_end",
        None,
        {
            "command": command,
            "container_id": container_id_or_name,
            "stdout": result.get("stdout"),
            "stderr": result.get("stderr"),
            "exit_code": result.get("exit_code"),
        },
    )
    return _json(result)


@mcp.tool()
@logged_tool(RUN_CTX, "container_list_files")
def container_list_files(container_id_or_name: str, path: str = "/workspace") -> str:
    """
    List files in a path inside a specific running container.
    """
    ready, msg = _docker_ready()
    if not ready:
        return _json({"ok": False, "error": msg})

    target_path = _resolve_container_path(path)
    if target_path.rstrip("/") == "/workspace":
        files, mtimes = _list_container_files(container_id_or_name)
        return _json(
            {"ok": True, "files": files, "mtimes": mtimes, "path": "/workspace"}
        )

    result = _exec_in_container(container_id_or_name, ["ls", "-la", target_path])
    return _json(result)


@mcp.tool()
@logged_tool(RUN_CTX, "container_read_file")
def container_read_file(container_id_or_name: str, path: str) -> str:
    """
    Read a text file inside a specific running container.
    """
    ready, msg = _docker_ready()
    if not ready:
        return _json({"ok": False, "error": msg})

    target_path = _resolve_container_path(path)
    result = _exec_in_container(container_id_or_name, ["cat", target_path])
    return _json(result)


@mcp.tool()
@logged_tool(RUN_CTX, "container_write_file")
def container_write_file(
    container_id_or_name: str,
    path: str,
    content: str,
    overwrite: bool = True,
) -> str:
    """
    Write a text file inside a specific running container.
    """
    ready, msg = _docker_ready()
    if not ready:
        return _json({"ok": False, "error": msg})

    target_path = _resolve_container_path(path)
    if not overwrite:
        exists = _exec_in_container(container_id_or_name, ["test", "-e", target_path])
        if exists.get("ok"):
            return _json({"ok": False, "error": "File already exists"})

    _exec_in_container(
        container_id_or_name, ["mkdir", "-p", os.path.dirname(target_path)]
    )

    safe_path = shlex.quote(target_path)
    try:
        result = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                container_id_or_name,
                "sh",
                "-c",
                f"cat > {safe_path}",
            ],
            input=content,
            text=True,
            capture_output=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return _json(
            {
                "ok": False,
                "stdout": "",
                "stderr": "Command timed out",
                "exit_code": None,
            }
        )
    except Exception as e:
        return _json({"ok": False, "stdout": "", "stderr": str(e), "exit_code": None})

    return _json(
        {
            "ok": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
        }
    )


@mcp.tool()
@logged_tool(RUN_CTX, "docker_repl_export")
def docker_repl_export(
    session_id: str,
    sources: list[str] | None = None,
    dest_dir: str = ".",
    overwrite: bool = False,
    preserve_paths: bool = True,
    glob_patterns: list[str] | None = None,
    since_last_run: bool = False,
    ignore: list[str] | None = None,
) -> str:
    """
    Export files/directories from the Docker REPL workspace to the host workspace.
    """
    repl = _SESSIONS.get(session_id)
    if not repl:
        return _json({"ok": False, "error": f"Unknown session_id: {session_id}"})

    if not sources and not glob_patterns and not since_last_run:
        return _json({"ok": False, "error": "No sources specified"})

    ignore = ignore or [
        "__pycache__",
        "*.pyc",
        ".DS_Store",
        ".git",
        ".ipynb_checkpoints",
    ]
    dest_root = _safe_host_path(dest_dir)
    os.makedirs(dest_root, exist_ok=True)

    base_dir = repl.temp_dir
    candidates: list[str] = []

    if sources:
        for src in sources:
            candidates.append(src)

    if glob_patterns:
        for pattern in glob_patterns:
            candidates.extend(
                [
                    os.path.relpath(path, base_dir)
                    for path in glob.glob(
                        os.path.join(base_dir, pattern), recursive=True
                    )
                ]
            )

    if since_last_run:
        last_ts = _SESSION_EXPORT_STATE.get(session_id, 0)
        for root, dirs, files in os.walk(base_dir):
            rel_root = os.path.relpath(root, base_dir)
            for name in list(dirs):
                if any(fnmatch.fnmatch(name, pat) for pat in ignore):
                    dirs.remove(name)
            for name in files:
                rel_path = os.path.normpath(os.path.join(rel_root, name))
                if any(fnmatch.fnmatch(name, pat) for pat in ignore):
                    continue
                full_path = os.path.join(root, name)
                if os.path.getmtime(full_path) > last_ts:
                    candidates.append(rel_path)

    # normalize
    norm_candidates = []
    for rel in candidates:
        rel = rel.lstrip("/\\")
        if rel in {".", ""}:
            continue
        if any(fnmatch.fnmatch(os.path.basename(rel), pat) for pat in ignore):
            continue
        norm_candidates.append(rel)

    seen = set()
    manifest = []
    for rel in norm_candidates:
        if rel in seen:
            continue
        seen.add(rel)
        src_path = _safe_path(base_dir, rel)
        if not os.path.exists(src_path):
            manifest.append({"source": rel, "status": "missing"})
            continue

        if preserve_paths:
            dest_path = os.path.join(dest_root, rel)
        else:
            dest_path = os.path.join(dest_root, os.path.basename(rel))

        if os.path.exists(dest_path) and not overwrite:
            manifest.append({"source": rel, "dest": dest_path, "status": "exists"})
            continue

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        try:
            if os.path.isdir(src_path):
                if os.path.exists(dest_path) and overwrite:
                    shutil.rmtree(dest_path)
                shutil.copytree(src_path, dest_path)
            else:
                shutil.copy2(src_path, dest_path)
            manifest.append({"source": rel, "dest": dest_path, "status": "copied"})
        except Exception as e:
            manifest.append(
                {"source": rel, "dest": dest_path, "status": "error", "error": str(e)}
            )

    _SESSION_EXPORT_STATE[session_id] = time.time()

    return _json({"ok": True, "manifest": manifest})


@mcp.tool()
@logged_tool(RUN_CTX, "container_export")
def container_export(
    container_id_or_name: str,
    sources: list[str] | None = None,
    dest_dir: str = ".",
    overwrite: bool = False,
    preserve_paths: bool = True,
    glob_patterns: list[str] | None = None,
    since_last_run: bool = False,
    ignore: list[str] | None = None,
) -> str:
    """
    Export files/directories from a running container /workspace to the host workspace.
    """
    ready, msg = _docker_ready()
    if not ready:
        return _json({"ok": False, "error": msg})

    if not sources and not glob_patterns and not since_last_run:
        return _json({"ok": False, "error": "No sources specified"})

    ignore = ignore or [
        "__pycache__",
        "*.pyc",
        ".DS_Store",
        ".git",
        ".ipynb_checkpoints",
    ]
    dest_root = _safe_host_path(dest_dir)
    os.makedirs(dest_root, exist_ok=True)

    candidates: list[str] = []
    if sources:
        candidates.extend(sources)

    files, mtimes = _list_container_files(container_id_or_name)

    if glob_patterns:
        for path in files:
            if any(fnmatch.fnmatch(path, pattern) for pattern in glob_patterns):
                candidates.append(path)

    if since_last_run:
        last_ts = _CONTAINER_EXPORT_STATE.get(container_id_or_name, 0)
        for path in files:
            mtime = mtimes.get(path, 0)
            if mtime > last_ts:
                candidates.append(path)

    norm_candidates = []
    for rel in candidates:
        rel = rel.lstrip("/\\")
        if rel in {".", ""}:
            continue
        if any(fnmatch.fnmatch(os.path.basename(rel), pat) for pat in ignore):
            continue
        norm_candidates.append(rel)

    seen = set()
    manifest = []
    for rel in norm_candidates:
        if rel in seen:
            continue
        seen.add(rel)

        if preserve_paths:
            dest_path = os.path.join(dest_root, rel)
        else:
            dest_path = os.path.join(dest_root, os.path.basename(rel))

        if os.path.exists(dest_path) and not overwrite:
            manifest.append({"source": rel, "dest": dest_path, "status": "exists"})
            continue

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        container_path = _resolve_container_path(rel)
        try:
            result = subprocess.run(
                ["docker", "cp", f"{container_id_or_name}:{container_path}", dest_path],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                manifest.append(
                    {
                        "source": rel,
                        "dest": dest_path,
                        "status": "error",
                        "error": result.stderr or result.stdout,
                    }
                )
                continue
            manifest.append({"source": rel, "dest": dest_path, "status": "copied"})
        except Exception as e:
            manifest.append(
                {"source": rel, "dest": dest_path, "status": "error", "error": str(e)}
            )

    _CONTAINER_EXPORT_STATE[container_id_or_name] = time.time()
    return _json({"ok": True, "manifest": manifest})


@mcp.tool()
@logged_tool(RUN_CTX, "docker_upload_local_file")
def docker_upload_local_file(
    session_id: str,
    local_relative_path: str,
    container_relative_path: str | None = None,
) -> str:
    """
    Upload a local workspace file into the container workspace.

    Output JSON contains: ok, host_path, container_path.
    """
    repl = _SESSIONS.get(session_id)
    if not repl:
        return _json({"ok": False, "error": f"Unknown session_id: {session_id}"})

    try:
        local_abs_path = _safe_path(BASE_DIR, local_relative_path)
    except Exception as e:
        return _json({"ok": False, "error": str(e)})

    if not os.path.exists(local_abs_path):
        return _json({"ok": False, "error": "Local file not found"})

    target_rel = container_relative_path or os.path.basename(local_abs_path)
    try:
        container_abs_path = _safe_path(repl.temp_dir, target_rel)
    except Exception as e:
        return _json({"ok": False, "error": str(e)})

    os.makedirs(os.path.dirname(container_abs_path), exist_ok=True)
    with open(local_abs_path, "r", encoding="utf-8") as f:
        content = f.read()

    with open(container_abs_path, "w", encoding="utf-8") as f:
        f.write(content)

    container_path = os.path.join("/workspace", target_rel.lstrip("/\\"))
    return _json(
        {"ok": True, "host_path": local_abs_path, "container_path": container_path}
    )


@mcp.tool()
@logged_tool(RUN_CTX, "stop_docker_repl_session")
def stop_docker_repl_session(session_id: str) -> str:
    """
    Stop a Docker REPL session and clean up its resources.
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
        {"env": "docker", "session_id": session_id, "container_id": repl.container_id},
    )

    return _json({"ok": True, "message": f"Stopped session {session_id}"})


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
